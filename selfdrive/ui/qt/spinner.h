#include <array>

#include <QLabel>
#include <QPixmap>
#include <QProgressBar>
#include <QSocketNotifier>
#include <QVariantAnimation>
#include <QWidget>

constexpr int spinner_fps = 30;
constexpr QSize spinner_size = QSize(360, 360);

class BodyAnimation;  // Forward declaration

class Spinner : public QWidget {
  Q_OBJECT

public:
  explicit Spinner(QWidget *parent = nullptr);
  void setAwake(bool awake);

private slots:
  void update(int n);

private:
  QLabel *text;
  QProgressBar *progress_bar;
  QSocketNotifier *notifier;
  BodyAnimation *body_animation;  // Replace TrackWidget with BodyAnimation
};
