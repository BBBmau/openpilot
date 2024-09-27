#pragma once

#include <QPainter>
#include "selfdrive/ui/ui.h"

class ModelRenderer : public QObject {
  Q_OBJECT

  public:
   ModelRenderer();
   void drawLaneLines(SubMaster &sm, const UIScene &scene, QLinearGradient &bg);

};