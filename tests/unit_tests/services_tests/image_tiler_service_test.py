from PIL import Image

from app.infrastructure.services.image_tiler_service import ImageTilerService


def test_tile_returns_single_tile_for_small_image():
    service = ImageTilerService(tile_size=640, overlap=100)
    image = Image.new("RGB", (320, 240))

    tiles = list(service.tile(image))

    assert len(tiles) == 1
    tile = tiles[0]
    assert tile.x == 0
    assert tile.y == 0
    assert tile.width == 320
    assert tile.height == 240


def test_tile_covers_large_image_with_overlap_and_border_alignment():
    service = ImageTilerService(tile_size=640, overlap=100)
    image = Image.new("RGB", (1200, 900))

    tiles = list(service.tile(image))

    positions = {(tile.x, tile.y) for tile in tiles}
    assert positions == {
        (0, 0),
        (540, 0),
        (560, 0),
        (0, 260),
        (540, 260),
        (560, 260),
    }

    for tile in tiles:
        assert tile.width == 640
        assert tile.height == 640


def test_shift_predictions_offsets_bbox_without_mutating_input():
    service = ImageTilerService()
    predictions = [
        {"class": "cat", "bbox": [1, 2, 11, 12], "confidence": 0.9},
        {"class": "no_bbox", "confidence": 0.5},
    ]

    shifted = service.shift_predictions(predictions, offset_x=100, offset_y=200)

    assert shifted[0]["bbox"] == [101, 202, 111, 212]
    assert shifted[0]["class"] == "cat"
    assert shifted[1] == {"class": "no_bbox", "confidence": 0.5}
    assert predictions[0]["bbox"] == [1, 2, 11, 12]


def test_merge_predictions_flattens_list_of_prediction_lists():
    service = ImageTilerService()

    merged = service.merge_predictions([
        [{"class": "cat"}],
        [{"class": "dog"}, {"class": "bird"}],
        [],
    ])

    assert merged == [
        {"class": "cat"},
        {"class": "dog"},
        {"class": "bird"},
    ]
