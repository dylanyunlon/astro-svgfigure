/**
 * geometry/index.ts — public API for the geometry module
 *
 * AT bundle coverage: Geometry, GeometryAttribute, BoxGeometry,
 * PlaneGeometry, SphereGeometry, CylinderGeometry  (6 of 14 AT geometry classes)
 */

export { Geometry }          from './Geometry';
export type { BoundingBox, DrawRange } from './Geometry';

export { GeometryAttribute } from './GeometryAttribute';
export type { AttributeArray } from './GeometryAttribute';

export { BoxGeometry }       from './BoxGeometry';
export { PlaneGeometry }     from './PlaneGeometry';
export { SphereGeometry }    from './SphereGeometry';
export { CylinderGeometry }  from './CylinderGeometry';
