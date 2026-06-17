import { Container } from '../../upstream/pixijs-engine/src/scene/container/Container';
import { Graphics }  from '../../upstream/pixijs-engine/src/scene/graphics/shared/Graphics';
import { Text }      from '../../upstream/pixijs-engine/src/scene/text/Text';
import { TextStyle } from '../../upstream/pixijs-engine/src/scene/text/TextStyle';

export function createTooltip(stage: Container) {
  const c = new Container();
  c.visible = false;

  const bg  = new Graphics();
  const txt = new Text({
    text: '',
    style: new TextStyle({ fontSize: 11, fill: 0xffffff }),
  });

  c.addChild(bg, txt);
  stage.addChild(c);

  return {
    show(x: number, y: number, label: string, species: string) {
      txt.text = label + ' [' + species + ']';
      bg.clear();
      bg.roundRect(-4, -2, txt.width + 12, txt.height + 6, 3);
      bg.fill({ color: 0x2d2d2d, alpha: 0.95 });
      c.position.set(x + 10, y - 25);
      c.visible = true;
    },
    hide() {
      c.visible = false;
    },
  };
}
