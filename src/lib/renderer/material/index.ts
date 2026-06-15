/**
 * material/index.ts — AT Material 系统入口
 *
 * 导出三个核心材质类：
 *   Material      基础材质类 (program, uniforms, blending, depthTest, depthWrite, side)
 *   PBRMaterial   物理基础渲染 (Cook-Torrance BRDF, AT PBR/ATPBR/RoomPBR 对齐)
 *   CellMaterial  Cell 专用材质 (species 驱动 shader 选择, xiaodi_options_table 参数)
 */

export { Material } from './Material.js';
export type { BlendingMode, SideMode, UniformValue, TextureDescriptor } from './Material.js';

export { PBRMaterial, PBR_VERT_SRC, PBR_FRAG_SRC, hexToRGB } from './PBRMaterial.js';
export type { PBRMaterialOptions } from './PBRMaterial.js';

export {
  CellMaterial,
  CELL_VERT_SRC,
  SPECIES_SHADER_MAP,
  SPECIES_DEFAULTS,
  getSpeciesFragSource,
  FRAG_CIL_EYE,
  FRAG_CIL_BOLT,
  FRAG_CIL_VECTOR,
  FRAG_DEFAULT,
} from './CellMaterial.js';
export type { CellSpecies, CellMaterialUniforms } from './CellMaterial.js';
