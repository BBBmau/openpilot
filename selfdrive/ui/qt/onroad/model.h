#pragma once

#include <QPainter>
#include <QWidget>

#include "selfdrive/ui/ui.h"

class ModelRenderer : public QWidget {
  Q_OBJECT

  public:
   ModelRenderer(QWidget *parent = 0) : QWidget(parent) {}
   void drawLaneLines(SubMaster &sm, const UIScene &scene, QLinearGradient &bg);

};