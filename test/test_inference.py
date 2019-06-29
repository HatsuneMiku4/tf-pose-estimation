from itertools import product

import pytest

from run import *

image_choices = ['../images/p1.jpg', ]
model_choices = ['cmu', 'mobilenet_thin', 'mobilenet_v2_large', 'mobilenet_v2_small', ]
resize_choices = ['0x0', '432x368', '656x368', '1312x736', ]
resize_out_ratio_choices = [1.0, 4.0, ]


@pytest.mark.parametrize('image, model, resize, resize_out_ratio', product(
    image_choices, model_choices, resize_choices, resize_out_ratio_choices))
def test_single_run(image='../images/p1.jpg', model='cmu', resize='0x0', resize_out_ratio=4.0):
    """
    tf-pose-estimation run
    :param image: path to the image to be processed
    :param model: cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small
    :param resize: if provided, resize images before they are processed. Recommends : 432x368 or 656x368 or 1312x736
    :param resize_out_ratio: if provided, resize heatmaps before they are post-processed.
    :return: None
    """

    w, h = model_wh(resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    image = common.read_imgfile(image, None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % image)
        sys.exit(-1)

    t = time.time()
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (image, elapsed))

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        a = fig.add_subplot(2, 2, 1)
        a.set_title('Result')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

        # show network output
        a = fig.add_subplot(2, 2, 2)
        plt.imshow(bgimg, alpha=0.5)
        tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        tmp2 = e.pafMat.transpose((2, 0, 1))
        tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
        tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

        a = fig.add_subplot(2, 2, 3)
        a.set_title('Vectormap-x')
        # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        a = fig.add_subplot(2, 2, 4)
        a.set_title('Vectormap-y')
        # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()
        plt.show()
    except Exception as e:
        logger.warning('matplitlib error, %s' % e)
        cv2.imshow('result', image)
        cv2.waitKey()
