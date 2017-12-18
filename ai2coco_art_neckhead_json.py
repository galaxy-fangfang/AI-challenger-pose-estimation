#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Author: fangfang

from __future__ import print_function
import os
import argparse
import json
from PIL import Image

def convert2coco(args):
    ai = json.load(open(args.keypoint_json,'r'))
    imgdir = args.img_dir
    print(imgdir)
    coco = dict()
    coco[u'info'] = {u'description':u'AI challenger keypoint in mscoco format'}
    coco[u'images'] = list()
    coco[u'annotations'] = list()
    coco[u'neck'] = list()
    cnt = 0
    #debug = []
    for ind,sample in enumerate(ai):
        #print('sample_id:',sample['image_id'+'.jpg'])
        img = Image.open(os.path.join(imgdir,sample['image_id']+'.jpg'))
        width, height = img.size

        coco_img = {}
        coco_img[u'license'] = 0
        coco_img[u'file_name'] = sample['image_id']
        coco_img[u'width'] = width
        coco_img[u'height'] = height
        coco_img[u'data_captured'] = 0
        coco_img[u'coco_url'] = sample['url']
        coco_img[u'flickr_url'] = sample['url']
        #in coco,the images->id is the number in file_name,but we
        coco_img['id'] = ind

        #cnt = 0
        # num of persons
        for human in enumerate(sample['keypoint_annotations']):
            num_keypoint = 0
            cocokey = list()
            v = list()

            # indx of keypoint_list
            for i in range(len(sample['keypoint_annotations'][human[1]])):
                #the first i = 2
                if (i+1)%3 == 0:#i = 2,5,8,11,14,17,20,23,26,29,31,35,38,41
                    if sample['keypoint_annotations'][human[1]][i] ==3:
                        v.append(0)
                    elif sample['keypoint_annotations'][human[1]][i] ==2 :
                        if i != 41:
                            num_keypoint += 1
                        v.append(1)
                    else:
                        if i != 41:
                            num_keypoint += 1
                        v.append(2)


			# in COCO,v = 0,not labeled;v = 1;labeled but invisible;v = 2,labeled and visible
			#in AI_challenge,v = 3,v=2,v=1
			# nose
			#if sample['keypoint_annotations'][human[1]][38] == 3:
            #if ind == 3873:
            #    print(cocokey)
            #    print(sample['keypoint_annotations'][human[1]])
             #   print('start')

            #cocokey[0:3] = [sample['keypoint_annotations'][human[1]][36],
            #               sample['keypoint_annotations'][human[1]][37],v[12]]
            
            #nose<-----neck(mpii)
            cocokey[0:3] = [sample['keypoint_annotations'][human[1]][39],
                            sample['keypoint_annotations'][human[1]][40],v[13]]
            #left_eye<------head(mpii)
            cocokey[3:6] = [sample['keypoint_annotations'][human[1]][36],
                            sample['keypoint_annotations'][human[1]][37],v[12]]
            # right_eye
            cocokey[6:9] = [0, 0, 0]
            # left_ear
            cocokey[9:12] = [0, 0, 0]
            # right_ear
            cocokey[12:15] = [0, 0, 0]
            # left_shoulder
            cocokey[15:18] = [sample['keypoint_annotations'][human[1]][9],
                              sample['keypoint_annotations'][human[1]][10],v[3]]
            # right_shoulder
            cocokey[18:21] = [sample['keypoint_annotations'][human[1]][0],
                              sample['keypoint_annotations'][human[1]][1],v[0]]
            # left_elbow
            cocokey[21:24] = [sample['keypoint_annotations'][human[1]][12],
                              sample['keypoint_annotations'][human[1]][13],v[4]]
            # right_elbow
            cocokey[24:27] = [sample['keypoint_annotations'][human[1]][3],
                              sample['keypoint_annotations'][human[1]][4],v[1]]
            # left_wrist
            cocokey[27:30] = [sample['keypoint_annotations'][human[1]][15],
                              sample['keypoint_annotations'][human[1]][16],v[5]]
            # right_wrist
            cocokey[30:33] = [sample['keypoint_annotations'][human[1]][6],
                              sample['keypoint_annotations'][human[1]][7],v[2]]
            # left_hip
            cocokey[33:36] = [sample['keypoint_annotations'][human[1]][27],
                              sample['keypoint_annotations'][human[1]][28],v[9]]
            # right_hip
            cocokey[36:39] = [sample['keypoint_annotations'][human[1]][18],
                              sample['keypoint_annotations'][human[1]][19],v[6]]
            # left_knee
            cocokey[39:42] = [sample['keypoint_annotations'][human[1]][30],
                              sample['keypoint_annotations'][human[1]][31],v[10]]
            # right_knee
            cocokey[42:45] = [sample['keypoint_annotations'][human[1]][21],
                              sample['keypoint_annotations'][human[1]][22],v[7]]
            # left_ankle
            cocokey[45:48] = [sample['keypoint_annotations'][human[1]][33],
                              sample['keypoint_annotations'][human[1]][34],v[11]]
            # right_ankle
            cocokey[48:51] = [sample['keypoint_annotations'][human[1]][24],
                              sample['keypoint_annotations'][human[1]][25],v[8]]


            coco_anno = {}
            coco_anno[u'num_keypoints'] = num_keypoint
            coco_anno[u'image_id'] = ind
            # the number of all the persons
            coco_anno[u'id'] = cnt

            coco_anno[u'keypoints'] = cocokey
            #coco->annotations->bbox is (x,y,width,height);
            #while ai->human_annotations is(leftup,rightbottom)
            coco_anno[u'bbox'] = [sample['human_annotations'][human[1]][0],
                                  sample['human_annotations'][human[1]][1],
                                  sample['human_annotations'][human[1]][2] - sample['human_annotations'][human[1]][0],
                                  sample['human_annotations'][human[1]][3] - sample['human_annotations'][human[1]][1]]
            #coco->annotation->area is the area of the encoded mask
            #here we use the area of the bbox to denote the area
            coco_anno[u'area'] = coco_anno[u'bbox'][2] * coco_anno[u'bbox'][3]
            #0 single object 1 a crowd of objects
            coco_anno[u'iscrowd'] = 0
            coco_anno[u'segmentation'] = [[
                sample['human_annotations'][human[1]][0],
                sample['human_annotations'][human[1]][1],
                sample['human_annotations'][human[1]][2],
                sample['human_annotations'][human[1]][1],
                sample['human_annotations'][human[1]][2],
                sample['human_annotations'][human[1]][3],
                sample['human_annotations'][human[1]][0],
                sample['human_annotations'][human[1]][3]]]
            #add the information of neck
            coco_neck = []
            coco_neck = [sample['keypoint_annotations'][human[1]][39],
                             sample['keypoint_annotations'][human[1]][40],v[13]]


            coco[u'annotations'].append(coco_anno)
            #coco[u'neck'].append(coco_neck)
            cnt += 1

        #if ind == 3873:

            #break
        coco[u'images'].append(coco_img)

        print('{}/{}'.format(ind,len(ai)))

    output_file = os.path.join(os.path.dirname(args.keypoint_json),'coco_art_neckhead_'+os.path.basename(args.keypoint_json))
    with open(output_file,'w') as fid:
        json.dump(coco,fid)
    print('Saved to {}'.format(output_file))

def main(args):
    convert2coco(args)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Convert AI challenger keypoint annotations to mscoco format')
    parser.add_argument('--keypoint_json',default ='dataset/AI_challenge/train/keypoint_train_annotations_20170909.json',type = str,help = 'keypoint json file path')
    parser.add_argument('--img_dir',default ='dataset/AI_challenge/train/keypoint_train_images_20170902',type = str,help = 'description')
    #parser.add_argument('--output_dir',default = 'dataset/AI_challenge/train')
    args  = parser.parse_args()
    print(args)
    #print('hahah')
    main(args)

