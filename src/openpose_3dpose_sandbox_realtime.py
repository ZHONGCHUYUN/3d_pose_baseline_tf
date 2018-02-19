
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import data_utils
import viz
import re
import cameras
import json
import os
from predict_3dpose import create_model
import cv2
import imageio
import logging
FLAGS = tf.app.flags.FLAGS

order = [15, 12, 25, 26, 27, 17, 18, 19, 1, 2, 3, 6, 7, 8]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(_):
    done = []

    enc_in = np.zeros((1, 64))
    enc_in[0] = [0 for i in range(64)]

    actions = data_utils.define_actions(FLAGS.action)

    SUBJECT_IDS = [1, 5, 6, 7, 8, 9, 11]
    rcams = cameras.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(
        actions, FLAGS.data_dir)
    train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
        actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14)

    device_count = {"GPU": 0}
    png_lib = []
    with tf.Session(config=tf.ConfigProto(
            device_count=device_count,
            allow_soft_placement=True)) as sess:
        #plt.figure(3)
        batch_size = 128
        model = create_model(sess, actions, batch_size)

        while True:
            #logger.info("start reading data")
            #load json files
            json_files = os.listdir(openpose_output_dir)
            # check for other file types
            json_files = sorted([filename for filename in json_files if filename.endswith(".json")])

            for file_name in json_files:
                if not file_name in done:
                    print ("reading...... ", file_name)
                    try:
                        _file = os.path.join(openpose_output_dir, file_name)
                        if not os.path.isfile(_file): raise Exception("No file found!!, {0}".format(_file))
                        data = json.load(open(_file))
                        #take first person
                        _data = data["people"][0]["pose_keypoints"]
                        xy = []
                        #ignore confidence score
                        for o in range(0,len(_data),3):
                            xy.append(_data[o])
                            xy.append(_data[o+1])

                        print (xy)  
                        frame_indx = re.findall("(\d+)", file_name)
                        frame = int(frame_indx[0])
                        """
                        forward,back = ([] for _ in range(2))
                        _len = len(xy) # 36
                        # create array of parallel frames (-3<n>3)
                        # first n frames, get value of xy in postive lookahead frames(current frame + 3)
                        if frame in first_frame_block:
                            for forward_range in range(1,4):
                                forward += cache[frame+forward_range]

                        # last n frames, get value of xy in negative lookahead frames(current frame - 3)
                        elif frame in last_frame_block:
                            for back_range in range(1,4):
                                back += cache[frame-forward_range]
                        # between frames, get value of xy in bi-directional frames(current frame -+ 6)     
                        else:
                            for forward_range in range(1,7):
                                forward += cache[frame+forward_range]
                            for back_range in range(1,7):
                                back += cache[frame-forward_range]

                        # build frame range vector 
                        frames_joint_median = [0 for i in range(_len)]
                        # more info about mapping in src/data_utils.py

                        # for each 18joints*x,y  (x1,y1,x2,y2,...)~36 
                        for x in range(0,_len,2):
                            # set x and y
                            y = x+1
                            if frame in first_frame_block:
                                # get vector of n frames forward for x and y, incl. current frame
                                x_v = [xy[x], forward[x], forward[x+_len], forward[x+_len*2]]
                                y_v = [xy[y], forward[y], forward[y+_len], forward[y+_len*2]]
                            elif frame in last_frame_block:
                                # get vector of n frames back for x and y, incl. current frame
                                x_v =[xy[x], back[x], back[x+_len], back[x+_len*2]]
                                y_v =[xy[y], back[y], back[y+_len], back[y+_len*2]]
                            else:
                                # get vector of n frames forward/back for x and y, incl. current frame
                                # median value calc: find neighbor frames joint value and sorted them, use numpy median module
                                # frame[x1,y1,[x2,y2],..]frame[x1,y1,[x2,y2],...], frame[x1,y1,[x2,y2],..]
                                #                 ^---------------------|-------------------------^
                                x_v =[xy[x], forward[x], forward[x+_len], forward[x+_len*2], forward[x+_len*3],forward[x+_len*4], forward[x+_len*5],
                                        back[x], back[x+_len], back[x+_len*2], back[x+_len*3], back[x+_len*4], back[x+_len*5]]
                                y_v =[xy[y], forward[y], forward[y+_len], forward[y+_len*2], forward[y+_len*3],forward[y+_len*4], forward[y+_len*5],
                                        back[y], back[y+_len], back[y+_len*2], back[y+_len*3], back[y+_len*4], back[y+_len*5]]

                            # get median of vector
                            x_med = np.median(sorted(x_v))
                            y_med = np.median(sorted(y_v))

                            # holding frame drops for joint
                            if not x_med:
                                # allow fix from first frame
                                if frame:
                                    # get x from last frame
                                    x_med = smoothed[frame-1][x]
                            # if joint is hidden y
                            if not y_med:
                                # allow fix from first frame
                                if frame:
                                    # get y from last frame
                                    y_med = smoothed[frame-1][y]

                            logger.debug("old X {0} sorted neighbor {1} new X {2}".format(xy[x],sorted(x_v), x_med))
                            logger.debug("old Y {0} sorted neighbor {1} new Y {2}".format(xy[y],sorted(y_v), y_med))

                            # build new array of joint x and y value
                            frames_joint_median[x] = x_med 
                            frames_joint_median[x+1] = y_med

                        """
                        joints_array = np.zeros((1, 36))
                        joints_array[0] = [0 for i in range(36)]
                        for o in range(len(joints_array[0])):
                            #feed array with xy array
                            joints_array[0][o] = xy[o]
                        _data = joints_array[0]
                        # mapping all body parts or 3d-pose-baseline format
                        for i in range(len(order)):
                            for j in range(2):
                                # create encoder input
                                enc_in[0][order[i] * 2 + j] = _data[i * 2 + j]
                        for j in range(2):
                            # Hip
                            enc_in[0][0 * 2 + j] = (enc_in[0][1 * 2 + j] + enc_in[0][6 * 2 + j]) / 2
                            # Neck/Nose
                            enc_in[0][14 * 2 + j] = (enc_in[0][15 * 2 + j] + enc_in[0][12 * 2 + j]) / 2
                            # Thorax
                            enc_in[0][13 * 2 + j] = 2 * enc_in[0][12 * 2 + j] - enc_in[0][14 * 2 + j]

                        # set spine
                        spine_x = enc_in[0][24]
                        spine_y = enc_in[0][25]

                        enc_in = enc_in[:, dim_to_use_2d]
                        mu = data_mean_2d[dim_to_use_2d]
                        stddev = data_std_2d[dim_to_use_2d]
                        enc_in = np.divide((enc_in - mu), stddev)

                        dp = 1.0
                        dec_out = np.zeros((1, 48))
                        dec_out[0] = [0 for i in range(48)]
                        _, _, poses3d = model.step(sess, enc_in, dec_out, dp, isTraining=False)
                        all_poses_3d = []
                        enc_in = data_utils.unNormalizeData(enc_in, data_mean_2d, data_std_2d, dim_to_ignore_2d)
                        poses3d = data_utils.unNormalizeData(poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d)
                        gs1 = gridspec.GridSpec(1, 1)
                        gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
                        plt.axis('off')
                        all_poses_3d.append( poses3d )
                        enc_in, poses3d = map( np.vstack, [enc_in, all_poses_3d] )
                        subplot_idx, exidx = 1, 1
                        max = 0
                        min = 10000

                        for i in range(poses3d.shape[0]):
                            for j in range(32):
                                tmp = poses3d[i][j * 3 + 2]
                                poses3d[i][j * 3 + 2] = poses3d[i][j * 3 + 1]
                                poses3d[i][j * 3 + 1] = tmp
                                if poses3d[i][j * 3 + 2] > max:
                                    max = poses3d[i][j * 3 + 2]
                                if poses3d[i][j * 3 + 2] < min:
                                    min = poses3d[i][j * 3 + 2]

                        for i in range(poses3d.shape[0]):
                            for j in range(32):
                                poses3d[i][j * 3 + 2] = max - poses3d[i][j * 3 + 2] + min
                                poses3d[i][j * 3] += (spine_x - 630)
                                poses3d[i][j * 3 + 2] += (500 - spine_y)

                        # Plot 3d predictions
                        ax = plt.subplot(gs1[subplot_idx - 1], projection='3d')
                        ax.view_init(18, -70)    
                        logger.debug(np.min(poses3d))
                        if np.min(poses3d) < -1000 and frame != 0:
                            poses3d = before_pose

                        p3d = poses3d

                        viz.show3Dpose(p3d, ax, lcolor="#9b59b6", rcolor="#2ecc71")
                        before_pose = poses3d
                        pngName = 'png/test_{0}.png'.format(str(frame))
                        plt.savefig(pngName)
                        #plt.show()
                        img = cv2.imread(pngName,0)
                        rect_cpy = img.copy()
                        cv2.imshow('3d-pose-baseline', img)

                        done.append(file_name)
                    except:
                        pass



if __name__ == "__main__":

    openpose_output_dir = FLAGS.openpose
    
    level = {0:logging.ERROR,
             1:logging.WARNING,
             2:logging.INFO,
             3:logging.DEBUG}

    logger.setLevel(level[FLAGS.verbose])


    tf.app.run()