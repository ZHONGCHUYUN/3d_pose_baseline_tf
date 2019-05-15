import json 
import maya.cmds as cmds
import pymel.core as pm
import maya.OpenMaya as om
import math
import re
import os
threed_pose_baseline = "/3d-pose-baseline/"
openpose_images = "/3d-pose-baseline/test_images/"

#jnt parent mapping dict
jnt_mapping = { 'root': [{'jnt_11': ['jnt_1','jnt_6']},
                         {'jnt_13': ['jnt_17', 'jnt_25']}],
                #left leg from jnt 6 to 8
                "left_leg":[{"jnt_{0}".format(n):"jnt_{0}".format(n+1)} for n in range(6,8)],
                #right leg from jnt 1 to 3
                "right_leg":[{"jnt_{0}".format(n):"jnt_{0}".format(n+1)} for n in range(1,3)],
                #left arm from jnt 17 to 19
                "left_arm":[{"jnt_{0}".format(n):"jnt_{0}".format(n+1)} for n in range(17,19)],
                #right arm from jnt 17 to 19
                "right_arm":[{"jnt_{0}".format(n):"jnt_{0}".format(n+1)} for n in range(25,27)]}


        
def load_data(data):
    # jnts to ignore
    to_pass = [5,4,9,10,12]
    # locator driver grp
    driver_grp = cmds.group(n="drivers", em=True)
    for frame, jnt in data.iteritems():
        if not cmds.objExists("anim_joint"):
            anim_grp = cmds.group(n="anim_joint", em=True)
        for jnt_id, trans in jnt.iteritems():
            if not int(jnt_id) in to_pass:
                if not cmds.objExists("anim_jnt_driver_{0}".format(jnt_id)):
                    cmds.select(clear=True)
                    jnt = cmds.joint(n="jnt_{0}".format(jnt_id), relative=True)
                    cmds.setAttr("{0}.radius".format(jnt), 50)
                    cmds.setAttr("{0}.displayLocalAxis".format(jnt), 1)
                    # match same pos for first frame
                    cmds.move(trans["translate"][0],trans["translate"][1],trans["translate"][2], jnt)
                    cmds.parent(jnt, anim_grp)
                    # driver locator
                    driver = cmds.spaceLocator(n="anim_jnt_driver_{0}".format(jnt_id))
                    # drive jnt with animated locator frim frame 0
                    cmds.pointConstraint(driver, jnt)
                    cmds.parent(driver, driver_grp)
                # add trans anim values to driver locator
                cmds.setKeyframe("anim_jnt_driver_{0}".format(jnt_id), t=frame, v=trans["translate"][0], at='translateX')
                cmds.setKeyframe("anim_jnt_driver_{0}".format(jnt_id), t=frame, v=trans["translate"][1], at='translateY')
                cmds.setKeyframe("anim_jnt_driver_{0}".format(jnt_id), t=frame, v=trans["translate"][2], at='translateZ')

    # hacking 3d-pose-baseline coord. to maya 
    # FIX ME
    cmds.setAttr("drivers.rotateX", -110)

def parent_skeleton():
    for body_part, jnt_map in jnt_mapping.iteritems():
        for map_dict in jnt_map:
            for parent_jnt, child_jnt in map_dict.iteritems():
                if isinstance(child_jnt, list):
                    for child in child_jnt:
                        cmds.parent(child, parent_jnt)
                else:
                    cmds.parent(child_jnt,parent_jnt)
def get_rotate(p1, p2):
    punkt_a = om.MPoint(p1[0], p1[1], p1[2])
    punkt_b = om.MPoint(p2[0], p2[1], p2[2])
    rot_vector = punkt_a - punkt_b
    world = om.MVector(0, 1, 0)
    quat = om.MQuaternion(world, rot_vector, 1) 
    mat = om.MTransformationMatrix()
    util = om.MScriptUtil()
    util.createFromDouble(0, 0, 0)
    rot_i = util.asDoublePtr()
    mat.setRotation(rot_i, om.MTransformationMatrix.kXYZ)
    mat = mat.asMatrix() * quat.asMatrix()
    quat = om.MTransformationMatrix(mat).rotation()
    m_rotation = om.MVector(math.degrees(quat.asEulerRotation().x),
                                        math.degrees(quat.asEulerRotation().y),
                                        math.degrees(quat.asEulerRotation().z)
                                        )
                                        
    return (m_rotation[0],m_rotation[1],m_rotation[2])
                      
def set_orient(data):
    for frame, jnt in data.iteritems():
        cmds.currentTime(int(frame))
        for body_part, jnt_map in jnt_mapping.iteritems():
            for map_dict in jnt_map:
                for parent_jnt, child_jnt in map_dict.iteritems():
                    if not isinstance(child_jnt, list):
                        p1 = cmds.xform(parent_jnt, q=True, t=True, ws=True)
                        p2 = cmds.xform(child_jnt, q=True, t=True, ws=True)
                        rotation = get_rotate(p1,p2)
                        cmds.setKeyframe(parent_jnt, t=frame, v=rotation[0], at='rotateX')
                        cmds.setKeyframe(parent_jnt, t=frame, v=rotation[1], at='rotateY')
                        cmds.setKeyframe(parent_jnt, t=frame, v=rotation[2], at='rotateZ')

threed_pose_output_json = os.path.join(threed_pose_baseline, "maya/3d_data.json")

with open(threed_pose_output_json) as json_data:
    data = json.load(json_data)
    # loaded data format:
    # frames x jnts x (x,y,z)
    # {frame<n>:[jnt<n>:"translate":[x,y,z], jnt<n>:"translate":[x,y,z], ...], frame<n>:...}
load_data(data)
parent_skeleton()
set_orient(data)
#set imageplane
#convert openpose --write_images output to valid padding for maya
for image_file in os.listdir(openpose_images):
    file_path = os.path.join(openpose_images, image_file)
    frame_idx = int(re.findall("(\d+)", image_file)[-1]) 
    os.rename(file_path, os.path.join(threed_pose_baseline, "maya/image_plane/image.{0}.png".format(frame_idx)))
print "use", os.path.join(threed_pose_baseline, "maya/image_plane/"), " for imageplane."

