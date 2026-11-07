import numpy as np

from matching.learned import list_images, resize_scale, to_colmap_keypoints


def test_resize_scale_shrinks_the_longer_side_and_never_enlarges():
    assert resize_scale(6048, 4032, 3024) == 0.5
    assert resize_scale(4032, 6048, 3024) == 0.5
    assert resize_scale(800, 600, 1600) == 1.0


def test_keypoints_map_back_to_colmap_pixel_convention():
    # The network's first pixel centre is 0; COLMAP's is 0.5. At scale 0.5 pixel 10 covers full-size 20 to 22.
    xy = to_colmap_keypoints(np.array([[0.0, 10.0]]), 0.5)
    assert xy.dtype == np.float32
    assert np.allclose(xy, [[1.0, 21.0]])


def test_keypoints_at_full_size_only_get_the_half_pixel_shift():
    assert np.allclose(to_colmap_keypoints(np.array([[3.0, 4.0]]), 1.0), [[3.5, 4.5]])


def test_list_images_is_sorted_and_ignores_other_files(tmp_path):
    for name in ("b.JPG", "a.jpg", "notes.txt", "c.png"):
        (tmp_path / name).write_bytes(b"")
    assert list_images(tmp_path) == ["a.jpg", "b.JPG", "c.png"]
