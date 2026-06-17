export interface CellState {
  id: string;
  species: string;
  x: number;
  y: number;
  w: number;
  h: number;
  opacity: number;
}

export interface AppState {
  cells: Map<string, CellState>;
  epoch: number;
  rate: number;
  selectedId: string | null;
  zoom: number;
}

export const state: AppState = {
  cells: new Map(),
  epoch: 0,
  rate: 1,
  selectedId: null,
  zoom: 1,
};

export function selectCell(id: string) {
  state.selectedId = id;
}

export function clearSelection() {
  state.selectedId = null;
}
