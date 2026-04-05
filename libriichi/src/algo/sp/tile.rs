use crate::tile::Tile;

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct DiscardTile {
    pub(super) tile: Tile,
    pub(super) shanten_diff: i8,
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct DrawTile {
    pub(super) tile: Tile,
    pub(super) count: u8,
    pub(super) shanten_diff: i8,
}

#[derive(Debug, Default)]
pub(super) struct DrawTileScan {
    pub(super) draw_tiles: tinyvec::ArrayVec<[DrawTile; 37]>,
    pub(super) sum_required_tiles: u8,
    pub(super) sum_left_tiles: u8,
}

#[derive(Debug, Default)]
pub(super) struct RequiredDrawTileScan {
    pub(super) draw_scan: DrawTileScan,
    pub(super) required_tiles: tinyvec::ArrayVec<[RequiredTile; 34]>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RequiredTile {
    pub tile: Tile,
    pub count: u8,
}
