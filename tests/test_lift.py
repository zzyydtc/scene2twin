import numpy as np

from semantics.lift import assign_mask_ids, cluster_instances


def square_mask(h, w, y0, y1, x0, x1):
    seg = np.zeros((h, w), dtype=bool)
    seg[y0:y1, x0:x1] = True
    return {"segmentation": seg, "area": int(seg.sum())}


def test_smallest_containing_mask_wins():
    big = square_mask(100, 100, 0, 40, 0, 40)  # 1600 px, below the 25% limit
    small = square_mask(100, 100, 10, 20, 10, 20)
    xy = np.array([[15.0, 15.0], [30.0, 30.0], [80.0, 80.0]])
    ids = assign_mask_ids([big, small], (100, 100), xy)
    assert ids.tolist() == [1, 0, -1]


def test_room_sized_masks_are_ignored():
    wall = square_mask(100, 100, 0, 100, 0, 60)  # 60% of the image
    ids = assign_mask_ids([wall], (100, 100), np.array([[10.0, 10.0]]))
    assert ids.tolist() == [-1]


def test_points_sharing_masks_across_views_form_instances():
    # Two objects (points 0-5 and 6-11) seen in 3 views with arbitrary per-view mask ids.
    pts = np.arange(12)
    views = []
    for offset in (0, 7, 20):
        ids = np.where(pts < 6, offset, offset + 1)
        views.append((pts, ids))
    inst = cluster_instances(views, 12, min_size=3)
    assert len({*inst.labels[:6]}) == 1 and len({*inst.labels[6:]}) == 1
    assert inst.labels[0] != inst.labels[6]
    assert all(a > 0.9 for a in inst.edge_affinity.values())


def test_single_view_agreement_is_not_enough():
    pts = np.arange(8)
    inst = cluster_instances([(pts, np.zeros(8, dtype=int))], 8, min_views=2, min_size=3)
    assert (inst.labels == -1).all()


def test_inconsistent_views_lower_the_link():
    # Point 4 joins the group in only one of three views, so it must not be linked.
    pts = np.arange(5)
    ids_same = np.array([0, 0, 0, 0, 1])
    ids_joined = np.array([0, 0, 0, 0, 0])
    views = [(pts, ids_same), (pts, ids_joined), (pts, ids_same)]
    inst = cluster_instances(views, 5, min_views=2, min_size=3)
    assert inst.labels[4] == -1
    assert (inst.labels[:4] == inst.labels[0]).all() and inst.labels[0] >= 0
