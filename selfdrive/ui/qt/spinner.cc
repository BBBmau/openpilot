#include "selfdrive/ui/qt/spinner.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>

#include <QApplication>
#include <QGridLayout>
#include <QPainter>
#include <QString>
#include <QTransform>
#include <QMovie>
#include <QLabel>
#include <QProgressBar>

#include "system/hardware/hw.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"

class BodyAnimation : public QWidget {
  Q_OBJECT

public:
  BodyAnimation(QWidget *parent = nullptr) : QWidget(parent) {
    setAttribute(Qt::WA_OpaquePaintEvent);

    animation_label = new QLabel(this);
    animation_label->setAlignment(Qt::AlignCenter);

    awake_animation = new QMovie("../assets/body_awake.gif");
    asleep_animation = new QMovie("../assets/body_asleep.gif");

    animation_label->setMovie(awake_animation);
    awake_animation->start();

    setFixedSize(awake_animation->frameRect().size());
  }

  void setAwake(bool awake) {
    animation_label->movie()->stop();
    if (awake) {
      animation_label->setMovie(awake_animation);
      awake_animation->start();
    } else {
      animation_label->setMovie(asleep_animation);
      asleep_animation->start();
    }
  }

private:
  QLabel *animation_label;
  QMovie *awake_animation;
  QMovie *asleep_animation;
};

Spinner::Spinner(QWidget *parent) : QWidget(parent) {
  QGridLayout *main_layout = new QGridLayout(this);
  main_layout->setSpacing(0);
  main_layout->setMargin(200);

  body_animation = new BodyAnimation(this);
  main_layout->addWidget(body_animation, 0, 0, Qt::AlignHCenter | Qt::AlignVCenter);

  text = new QLabel();
  text->setWordWrap(true);
  text->setVisible(false);
  text->setAlignment(Qt::AlignCenter);
  main_layout->addWidget(text, 1, 0, Qt::AlignHCenter);

  progress_bar = new QProgressBar();
  progress_bar->setRange(5, 100);
  progress_bar->setTextVisible(false);
  progress_bar->setVisible(false);
  progress_bar->setFixedHeight(20);
  main_layout->addWidget(progress_bar, 1, 0, Qt::AlignHCenter);

  setStyleSheet(R"(
    Spinner {
      background-color: black;
    }
    QLabel {
      color: white;
      font-size: 80px;
      background-color: transparent;
    }
    QProgressBar {
      background-color: #373737;
      width: 1000px;
      border solid white;
      border-radius: 10px;
    }
    QProgressBar::chunk {
      border-radius: 10px;
      background-color: white;
    }
  )");

  notifier = new QSocketNotifier(fileno(stdin), QSocketNotifier::Read);
  QObject::connect(notifier, &QSocketNotifier::activated, this, &Spinner::update);
}

void Spinner::update(int n) {
  std::string line;
  std::getline(std::cin, line);

  if (line.length()) {
    bool number = std::all_of(line.begin(), line.end(), ::isdigit);
    text->setVisible(!number);
    progress_bar->setVisible(number);
    text->setText(QString::fromStdString(line));
    if (number) {
      progress_bar->setValue(std::stoi(line));
    }
  }
}

void Spinner::setAwake(bool awake) {
  if (body_animation) {
    body_animation->setAwake(awake);
  }
}

int main(int argc, char *argv[]) {
  initApp(argc, argv);
  QApplication a(argc, argv);
  Spinner spinner;
  setMainWindow(&spinner);
  return a.exec();
}
