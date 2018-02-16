import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import data_utils
import viz
import cameras
import json
import os
from predict_3dpose import create_model
import cv2
FLAGS = tf.app.flags.FLAGS
order = [15, 12, 25, 26, 27, 17, 18, 19, 1, 2, 3, 6, 7, 8]

openpose_output_dir = "/home/flyn/git/openpose/output/"
"""
json_files = os.listdir(openpose_output_dir)
for frame in range(len(json_files)):
    print(">>>>", frame)
    _file = os.path.join(openpose_output_dir, 'Animation Reference - Athletic Male Standard Walk_{0}_keypoints.json'.format(str(frame).zfill(12)))
    if not os.path.isfile(_file): raise Excepton("No file found!!, {0}".format(_file))
    data = json.load(open(_file))
    _data = data["people"][0]["pose_keypoints"]
    for o in range(2,len(_data),3):
        print (_data[o])
        #dd.append(_data[o])            
    #_data = dd
    break

"""



def main(_):
    #return
    enc_in = np.zeros((1, 64))
    enc_in[0] = [0 for i in range(64)]

    actions = data_utils.define_actions(FLAGS.action)

    SUBJECT_IDS = [1, 5, 6, 7, 8, 9, 11]
    rcams = cameras.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(
        actions, FLAGS.data_dir)
    train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
        actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14)

    device_count = {"GPU": 1}
    with tf.Session(config=tf.ConfigProto(
            device_count=device_count,
            allow_soft_placement=True)) as sess:
        batch_size = 128
        model = create_model(sess, actions, batch_size)
        json_files = os.listdir(openpose_output_dir)
        #before_pose = ""
        c = 0
        for frame in range(len(json_files)):
            #_file_name = json_files[frame]
            _file = os.path.join(openpose_output_dir, 'Animation Reference - Athletic Male Standard Walk_{0}_keypoints.json'.format(str(frame).zfill(12)))
            print(">>>>", _file)
            if not os.path.isfile(_file): raise Excepton("No file found!!, {0}".format(_file))
            data = json.load(open(_file))
            _data = data["people"][0]["pose_keypoints"]
            #print(_data)
            aa = []

            for o in range(0,len(_data),3):
                aa.append(_data[o])
                aa.append(_data[o+1])
            #print(aa)

            dd = np.zeros((1, 36))
            dd[0] = [0 for i in range(36)]
            for o in range(len(dd[0])):
                #print(o,"????",aa[o])
                dd[0][o] = aa[o]
                #dd[0][o+1] = aa[o+1]
            _data = dd[0]
            #print (_data.shape)
            print (_data)
            #break
            for i in range(len(order)):
                for j in range(2):
                    _or = order[i] * 2 + j
                    _or_result = _data[i * 2 + j]
                    #print (_or, " # ".zfill(j), _or_result)
                    enc_in[0][_or] = _or_result

            for j in range(2):
                # Hip
                hip = 0 * 2 + j
                hip_result = (enc_in[0][1 * 2 + j] + enc_in[0][6 * 2 + j]) / 2
                #print (hip, " > ".zfill(j), hip_result)
                enc_in[0][hip] = hip_result
                
                # Neck/Nose
                neck_nose = 14 * 2 + j
                neck_nose_result = (enc_in[0][15 * 2 + j] + enc_in[0][12 * 2 + j]) / 2
                #print (neck_nose, " > ".zfill(j), neck_nose_result)
                enc_in[0][neck_nose] = neck_nose_result
                # Thorax
                thorax = 13 * 2 + j
                thorax_result = 2 * enc_in[0][12 * 2 + j] - enc_in[0][14 * 2 + j]
                #print (thorax, " > ".zfill(j), thorax_result)
                enc_in[0][thorax] = thorax_result

            #print (enc_in)

            spine_x = enc_in[0][24]
            spine_y = enc_in[0][25]


            #print (enc_in.shape)

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
            ax3 = plt.subplot(gs1[subplot_idx - 1], projection='3d')

            print (">>>>>>>>>>>", np.min(poses3d),np.min(poses3d) < -1000)
            if np.min(poses3d) < -1000 and frame != 0:
                #before_pose = poses3d
                poses3d = before_pose
            print (enc_in)
            #break
            p3d = poses3d
            viz.show3Dpose(p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71")

            pngName = 'png/test{0}.png'.format(str(frame).zfill(12))
            plt.savefig(pngName)
            c += 1
            #if c == 2:
                #break
            #break

            before_pose = poses3d

if __name__ == "__main__":
    tf.app.run()