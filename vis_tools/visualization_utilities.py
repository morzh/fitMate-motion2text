import cv2
import numpy as np

def bbox_xyxy2xywh(bbox_xyxy):
    """Transform the bbox format from x1y1x2y2 to xywh.

    Args:
        bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])

    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    """
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0]
    bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1]

    return bbox_xywh


def map_joint_dict(joints: np.ndarray):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)

    return joints_dict


def vis_pose_result(image_name, pose_results, thickness, out_file):
    data_numpy = cv2.imread(image_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    h = data_numpy.shape[0]
    w = data_numpy.shape[1]

    # Plot
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = plt.subplot(1, 1, 1)
    bk = plt.imshow(data_numpy[:, :, ::-1])
    bk.set_zorder(-1)

    for i, dt in enumerate(pose_results[:]):
        dt_joints = np.array(dt['keypoints']).reshape(17, -1)
        joints_dict = map_joint_dict(dt_joints)

        # stick
        for k, link_pair in enumerate(chunhua_style.link_pairs):
            if k in range(11, 16):
                lw = thickness
            else:
                lw = thickness * 2

            line = mlines.Line2D(
                np.array([joints_dict[link_pair[0]][0],
                          joints_dict[link_pair[1]][0]]),
                np.array([joints_dict[link_pair[0]][1],
                          joints_dict[link_pair[1]][1]]),
                ls='-', lw=lw, alpha=1, color=link_pair[2], )
            line.set_zorder(0)
            ax.add_line(line)

        # black ring
        for k in range(dt_joints.shape[0]):
            if k in range(5):
                radius = thickness
            else:
                radius = thickness * 2

            circle = mpatches.Circle(tuple(dt_joints[k, :2]),
                                     radius=radius,
                                     ec='black',
                                     fc=chunhua_style.ring_color[k],
                                     alpha=1,
                                     linewidth=1)
            circle.set_zorder(1)
            ax.add_patch(circle)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(out_file, format='png', bbox_inches='tight', dpi=100)
    plt.close()


def draw_skeleton(img, pose_results, style, thickness=2):
    for i, dt in enumerate(pose_results[:]):
        dt_joints = pose_results[0][:, :2]
        joints_dict = map_joint_dict(dt_joints)

        # stick
        for k, link_pair in enumerate(style.link_pairs):
            if k in range(11, 16):
                lw = thickness
            else:
                lw = thickness * 2

            color_current = (255 * link_pair[2][2], 255 * link_pair[2][1], 255 * link_pair[2][0])
            img = cv2.line(img, joints_dict[link_pair[0]], joints_dict[link_pair[1]], color=color_current, thickness=lw)
            # img = cv2.line(img, (joints_dict[link_pair[0]][0], joints_dict[link_pair[1]][0]), (joints_dict[link_pair[0]][1], joints_dict[link_pair[1]][1]), color=link_pair[2], thickness=lw)

        # dark ring
        for k in range(dt_joints.shape[0]):
            if k in range(5):
                radius = thickness
            else:
                radius = thickness * 2

            img = cv2.circle(img, tuple(dt_joints[k].astype(int)), radius, (30, 30, 30), thickness)

        return img


def bboxes_dict2ndarray(persons_bboxes):
    bboxes = np.empty((0, 5))
    for index in range(len(persons_bboxes)):
        bboxes = np.vstack((bboxes, persons_bboxes[index]['bbox']))
    return bboxes


def get_largest_bbox(bboxes_xywh, threshold):

    number_boxes = len(bboxes_xywh)

    if number_boxes == 0:
        return None

    boxes_selected = []
    for index in range(number_boxes):
        if bboxes_xywh[index][4] > threshold:
            boxes_selected.append(bboxes_xywh[index])

    if len(boxes_selected) == 0:
        return None

    largest_bbox = boxes_selected[0]
    largest_bbox_area = largest_bbox[2] * largest_bbox[3]

    for index in range(1, len(boxes_selected)):
        current_bbox_area = boxes_selected[index][2] * boxes_selected[index][3]
        if current_bbox_area > largest_bbox_area:
            largest_bbox = boxes_selected[index]
            largest_bbox_area = current_bbox_area

    return largest_bbox


def draw_bbox(img, bbox_xywh, color=(255, 90, 90), thickness=2):
    bbox_xywh = bbox_xywh.astype(int)
    img = cv2.line(img, (bbox_xywh[0], bbox_xywh[1]), (bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1]), color=color, thickness=thickness)
    img = cv2.line(img, (bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1]), (bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]), color=color, thickness=thickness)
    img = cv2.line(img, (bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]), (bbox_xywh[0], bbox_xywh[1] + bbox_xywh[3]), color=color, thickness=thickness)
    img = cv2.line(img, (bbox_xywh[0], bbox_xywh[1] + bbox_xywh[3]), (bbox_xywh[0], bbox_xywh[1]), color=color, thickness=thickness)
    return img