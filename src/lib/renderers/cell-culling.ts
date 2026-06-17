import type { Container } from '../../upstream/pixijs-engine/src/scene/container/Container';

export function updateCulling(stage:Container,vp:{x:number,y:number,w:number,h:number}){stage.children.forEach(c=>{if(!c.getBounds)return;const b=c.getBounds();c.visible=!(b.x>vp.x+vp.w||b.x+b.width<vp.x||b.y>vp.y+vp.h||b.y+b.height<vp.y)})}
