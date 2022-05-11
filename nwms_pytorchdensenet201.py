# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2018. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

from __future__ import print_function, unicode_literals, absolute_import, division


__author__ = "WSH Munirah W Ahmad <wshmunirah@gmail.com>"
__copyright__ = "Apache 2 license. Made by Multimedia University Cytomine Team, Cyberjaya, Malaysia, http://cytomine.mmu.edu.my/"
__version__ = "1.0.0"

# Date created: 12 April 2022
# Date created (terminal): 04 April 2022

import sys
import numpy as np
import os
import cytomine
from shapely.geometry import shape, box, Polygon,Point
from shapely import wkt
from glob import glob
# from tifffile import imread

from cytomine import Cytomine, models, CytomineJob
from cytomine.models import Annotation, AnnotationTerm, AnnotationCollection, ImageInstanceCollection, Job, JobData, Project, ImageInstance, Property
from cytomine.models.ontology import Ontology, OntologyCollection, Term, RelationTerm, TermCollection

# from PIL import Image
import torch
from torchvision.models import DenseNet

# import matplotlib.pyplot as plt
import time
import cv2
import math

from argparse import ArgumentParser
import json
import logging
import shutil



__author__ = "WSH Munirah W Ahmad <wshmunirah@gmail.com>"
__version__ = "1.0.1"

def run(cyto_job, parameters):
    logging.info("----- Classify_NWMS_ER_pytorch_DenseNet201 v%s -----", __version__)
    logging.info("Entering run(cyto_job=%s, parameters=%s)", cyto_job, parameters)

    job = cyto_job.job
    user = job.userJob
    project = cyto_job.project


    # currentdir = os.path.dirname(__file__)
    terms = TermCollection().fetch_with_filter("project", project.id)
    # conn.job.update(status=Job.RUNNING, progress=1, statusComment="Terms collected...")
    print(terms)

    start_time=time.time()

    # ----- load network ----
    # model = "/models/3333nuclei_densenet201_best_model_100ep.pth"
    modelname = "/models/3333nuclei_densenet201_best_model_100ep.pth"
    gpuid = 0

    device = torch.device(gpuid if gpuid!=-2 and torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(modelname, map_location=lambda storage, loc: storage) #load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666

    model = DenseNet(growth_rate=checkpoint["growth_rate"], block_config=checkpoint["block_config"],
                    num_init_features=checkpoint["num_init_features"], bn_size=checkpoint["bn_size"],
                    drop_rate=checkpoint["drop_rate"], num_classes=checkpoint["num_classes"]).to(device)

    model.load_state_dict(checkpoint["model_dict"])
    model.eval()
    print("Model name: ",modelname)
    print(f"Model successfully loaded! Total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")
        
    #Select images to process
    images = ImageInstanceCollection().fetch_with_filter("project", project.id)
    # conn.job.update(status=Job.RUNNING, progress=2, statusComment="Images gathered...")
        
    # print('images id:',images)

    list_imgs = []
    if parameters.cytomine_id_images == 'all':
        for image in images:
            list_imgs.append(int(image.id))
    else:
        # list_imgs = [int(id_img) for id_img in parameters.cytomine_id_images.split(',')]
        list_imgs = parameters.cytomine_id_images
        list_imgs2 = list_imgs.split(',')
        
    print('Input param:', parameters.cytomine_id_images)
    print('Print list images:', list_imgs)
    print(type(list_imgs))
    # list_imgs2 = list_imgs.split(',')
    print(type(list_imgs2))
    print('Print list images2:', list_imgs2)
    # for id_image in list_imgs2:
    #     print(id_image) 


    working_path = os.path.join("tmp", str(job.id))

    if not os.path.exists(working_path):
        logging.info("Creating working directory: %s", working_path)
        os.makedirs(working_path)
    try:
        
        id_project=project.id   
        output_path = os.path.join(working_path, "densenet201_results.csv")
        f= open(output_path,"w+")
        #Go over images
        job.update(status=Job.RUNNING, progress=20, statusComment="Running DenseNet classification on image...")
        #for id_image in conn.monitor(list_imgs, prefix="Running PN classification on image", period=0.1):
        
        #Go over images
        # conn.job.update(status=Job.RUNNING, progress=10, statusComment="Running PN classification on image...")
        #for id_image in conn.monitor(list_imgs, prefix="Running PN classification on image", period=0.1):
        for id_image in list_imgs2:
            print('Current image:', id_image)
            roi_annotations = AnnotationCollection()
            roi_annotations.project = project.id
#             roi_annotations.term = parameters.cytomine_id_cell_term
            roi_annotations.image = id_image #conn.parameters.cytomine_id_image
            roi_annotations.job = parameters.cytomine_id_annotation_job
            roi_annotations.user = parameters.cytomine_id_user_job
            roi_annotations.showWKT = True
            roi_annotations.fetch()
            # print(roi_annotations)

            current_im = ImageInstance().fetch(id_image)
            # current_im2 = 

            start_prediction_time=time.time()
            predictions = []
            img_all = []
            pred_all = []
            pred_c0 = 0
            pred_c1 = 0
            pred_c2 = 0
            pred_c3 = 0

            f.write("AnnotationID;ImageID;ProjectID;JobID;TermID;UserID;Area;Perimeter;WKT \n")

            #Go over ROI in this image
            #for roi in conn.monitor(roi_annotations, prefix="Running detection on ROI", period=0.1):
            roi_numel=len(roi_annotations)
            x=range(1,roi_numel)
            increment=np.multiply(10000,x)
            
            print("----------------------------Classsifying Nucleus------------------------------")
            # --- run all nuclei for a job -----------
            for i, roi in enumerate(roi_annotations):
            # ----------------------------------------

            # --- run part of the nuclei for a job ---            
            # nuclei_processed=150047
            # remaining_roi=roi_numel-nuclei_processed            
            # for i in range(0,remaining_roi): 
            # # for j, i in range(151821,151822):      
            #     j=nuclei_processed-1+i
            #     print(j)   
            #     roi=roi_annotations[j]
            # ----------------------------------------
                
#                 print(i)
#                 print(roi)
                for inc in increment:
                    if i==inc:
                        shutil.rmtree(roi_path, ignore_errors=True)
                        import gc
                        gc.collect()
                        print("i==", inc)

                #Get Cytomine ROI coordinates for remapping to whole-slide
                #Cytomine cartesian coordinate system, (0,0) is bottom left corner                
#                 print("----------------------------Classsifying Nucleus------------------------------")
                roi_geometry = wkt.loads(roi.location)
                # print("ROI Geometry from Shapely: {}".format(roi_geometry))
                # print("ROI Bounds")
                # print(roi_geometry.bounds)
                minx=roi_geometry.bounds[0]
                miny=roi_geometry.bounds[3]
                #Dump ROI image into local PNG file
                roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/')
                # print(roi_path)
                roi_png_filename=os.path.join(roi_path+str(roi.id)+'.png')
                # job.update(status=Job.RUNNING, progress=20, statusComment="Classifying...")
                # job.update(status=Job.RUNNING, progress=20, statusComment=roi_png_filename)
                # print("roi_png_filename: %s" %roi_png_filename)
                
                roi.dump(dest_pattern=roi_png_filename,mask=True)

                im = cv2.cvtColor(cv2.imread(roi_png_filename),cv2.COLOR_BGR2RGB)
                im = cv2.resize(im,(224,224))
                im = im.reshape(-1,224,224,3)
                output = np.zeros((0,checkpoint["num_classes"]))
                arr_out_gpu = torch.from_numpy(im.transpose(0, 3, 1, 2)).type('torch.FloatTensor').to(device)
                output_batch = model(arr_out_gpu)
                output_batch = output_batch.detach().cpu().numpy()                
                output = np.append(output,output_batch,axis=0)
                pred_labels = np.argmax(output, axis=1)
                # pred_labels=[pred_labels]
                pred_all.append(pred_labels)
                # print(output)
                # print(output_batch)
                # print(type(pred_labels))
#                 print(pred_labels)
                # print(pred_all)

                if pred_labels[0]==0:
                    # print("Class 0: Negative")
                    id_terms=parameters.cytomine_id_c0_term
                    pred_c0=pred_c0+1
                    # roi.dump(dest_pattern=os.path.join(roi_path+'Class0/'+str(roi.id)+'.png'),alpha=True)
                elif pred_labels[0]==1:
                    # print("Class 1: Weak")
                    id_terms=parameters.cytomine_id_c1_term
                    pred_c1=pred_c1+1
                    # roi.dump(dest_pattern=os.path.join(roi_path+'Class1/'+str(roi.id)+'.png'),alpha=True)
                elif pred_labels[0]==2:
                    # print("Class 2: Moderate")
                    id_terms=parameters.cytomine_id_c2_term
                    pred_c2=pred_c2+1
                    # roi.dump(dest_pattern=os.path.join(roi_path+'Class2/'+str(roi.id)+'.png'),alpha=True)
                elif pred_labels[0]==3:
                    # print("Class 3: Strong")
                    id_terms=parameters.cytomine_id_c3_term
                    pred_c3=pred_c3+1
                    # roi.dump(dest_pattern=os.path.join(roi_path+'Class3/'+str(roi.id)+'.png'),alpha=True)


                cytomine_annotations = AnnotationCollection()
                annotation=roi_geometry
                cytomine_annotations.append(Annotation(location=annotation.wkt,#location=roi_geometry,
                                                       id_image=id_image,#conn.parameters.cytomine_id_image,
                                                       id_project=project.id,
                                                       id_terms=[id_terms]))
                print(".",end = '',flush=True)

                #Send Annotation Collection (for this ROI) to Cytomine server in one http request
                ca = cytomine_annotations.save()
                cytomine_annotations.project = project.id
                cytomine_annotations.job = job.id
                cytomine_annotations.user = user
                cytomine_annotations.showAlgo = True
                cytomine_annotations.showWKT = True
                cytomine_annotations.showMeta = True
                cytomine_annotations.showGIS = True
                cytomine_annotations.showTerm = True
                cytomine_annotations.annotation = True
                cytomine_annotations.fetch()
#                 print(cytomine_annotations)

            job.update(status=Job.RUNNING, progress=80, statusComment="Writing classification results on CSV...")
            for annotation in cytomine_annotations:
                # print(annotation.id)
                # f.write("{};{};{};{};{};{};{};{}\n".format(annotation.id,annotation.image,annotation.project,annotation.term,annotation.user,annotation.area,annotation.perimeter,annotation.location))
                f.write("{};{};{};{};{};{};{};{};{}\n".format(annotation.id,annotation.image,annotation.project,job.id,annotation.term,annotation.user,annotation.area,annotation.perimeter,annotation.location))
            
            job.update(status=Job.RUNNING, progress=90, statusComment="Generating scoring for whole-slide image(s)...")
            pred_all=[pred_c0, pred_c1, pred_c2, pred_c3]
            pred_positive_all=[pred_c1, pred_c2, pred_c3]
            print("pred_all:", pred_all)
            im_pred = np.argmax(pred_all)
            print("image prediction:", im_pred)
            pred_total=pred_c0+pred_c1+pred_c2+pred_c3
            print("pred_total:",pred_total)
            pred_positive=pred_c1+pred_c2+pred_c3
            print("pred_positive:",pred_positive)
            print("pred_positive_all:",pred_positive_all)
            print("pred_positive_max:",np.argmax(pred_positive_all))
            pred_positive_100=pred_positive/pred_total*100
            print("pred_positive_100:",pred_positive_100)

            if pred_positive_100 == 0:
                proportion_score = 0
            elif pred_positive_100 < 1:
                proportion_score = 1
            elif pred_positive_100 >= 1 and pred_positive_100 <= 10:
                proportion_score = 2
            elif pred_positive_100 > 10 and pred_positive_100 <= 33:
                proportion_score = 3
            elif pred_positive_100 > 33 and pred_positive_100 <= 66:
                proportion_score = 4
            elif pred_positive_100 > 66:
                proportion_score = 5

            if pred_positive_100 == 0:
                intensity_score = 0
            elif im_pred == 0:
                intensity_score = np.argmax(pred_positive_all)+1
            elif im_pred == 1:
                intensity_score = 1
            elif im_pred == 2:
                intensity_score = 2
            elif im_pred == 3:
                intensity_score = 3

            allred_score = proportion_score + intensity_score
            print('Proportion Score: ',proportion_score)
            print('Intensity Score: ',intensity_score)            
            print('Allred Score: ',allred_score)
            shutil.rmtree(roi_path, ignore_errors=True)
            
        end_time=time.time()
        print("Execution time: ",end_time-start_time)
        print("Prediction time: ",end_time-start_prediction_time)
        
        f.write(" \n")
        f.write("Class Prediction;Class 0 (Negative);Class 1 (Weak);Class 2 (Moderate);Class 3 (Strong);Total Prediction;Total Positive;Class Positive Max;Positive Percentage;Proportion Score;Intensity Score;Allred Score;Execution Time;Prediction Time \n")
        f.write("{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(im_pred,pred_c0,pred_c1,pred_c2,pred_c3,pred_total,pred_positive,np.argmax(pred_positive_all),pred_positive_100,proportion_score,intensity_score,allred_score,end_time-start_time,end_time-start_prediction_time))
        
        f.close()
        
        job.update(status=Job.RUNNING, progress=99, statusComment="Summarizing results...")
        job_data = JobData(job.id, "Generated File", "densenet201_results.csv").save()
        job_data.upload(output_path)

    finally:
        logging.info("Deleting folder %s", working_path)
        shutil.rmtree(working_path, ignore_errors=True)
        logging.debug("Leaving run()")

    job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.") 

if __name__ == "__main__":
    logging.debug("Command: %s", sys.argv)

    with cytomine.CytomineJob.from_cli(sys.argv) as cyto_job:
        run(cyto_job, cyto_job.parameters)


    # conn.job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.")

# if __name__ == "__main__":
#     main(sys.argv[1:])

    #with cytomine.CytomineJob.from_cli(sys.argv) as cyto_job:
        #run(cyto_job, cyto_job.parameters)



