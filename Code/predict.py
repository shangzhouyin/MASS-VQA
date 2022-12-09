import argparse
import json
import os.path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from torch.autograd import Variable
from tqdm import tqdm

import sys,os
from PyQt5 import QtGui,QtWidgets
from PyQt5 import QtCore

import models
from datasets import vqa_dataset


def predict_answers(model, loader, split):
    model.eval()
    predicted = []
    samples_ids = []

    tq = tqdm(loader)

    print("Evaluating...\n")

    for item in tq:
        v = item['visual']
        q = item['question']
        sample_id = item['sample_id']
        q_length = item['q_length']

        v = Variable(v.cuda(async=True))
        q = Variable(q.cuda(async=True))
        q_length = Variable(q_length.cuda(async=True))

        out = model(v, q, q_length)

        _, answer = out.data.cpu().max(dim=1)

        predicted.append(answer.view(-1))
        samples_ids.append(sample_id.view(-1).clone())

    predicted = list(torch.cat(predicted, dim=0))
    samples_ids = list(torch.cat(samples_ids, dim=0))

    print("Evaluation completed")

    return predicted, samples_ids


def create_submission(input_annotations, predicted, samples_ids, vocabs):
    answers = torch.FloatTensor(predicted)
    indexes = torch.IntTensor(samples_ids)
    ans_to_id = vocabs['answer']
    # need to translate answers ids into answers
    id_to_ans = {idx: ans for ans, idx in ans_to_id.items()}
    # sort based on index the predictions
    sort_index = np.argsort(indexes)
    sorted_answers = np.array(answers, dtype='int_')[sort_index]

    real_answers = []
    #for ans_id in sorted_answers:
    ans = id_to_ans[sorted_answers]
    real_answers.append(ans)

    # Integrity check
    assert len(input_annotations) == len(real_answers)

    submission = []
    for i in range(len(input_annotations)):
        pred = {}
        pred['image'] = input_annotations[i]['image']
        pred['answer'] = real_answers[i]
        submission.append(pred)

    return pred['answer']
    
    
class VQA_demo(QtWidgets.QWidget):
    
    def __init__(self):
        super(VQA_demo, self).__init__()     
        self.initUI()
            
    def initUI(self): 

        self.image_file_name = None
        self.question = None              
        
        self.l1=QtWidgets.QLabel()
        self.lbl_qstn=QtWidgets.QLabel()
        self.lbl_output=QtWidgets.QLabel()
        self.lbl_output.setAlignment(QtCore.Qt.AlignCenter)
        
        self.input_qstn = QtWidgets.QLineEdit()
         # Text edit
        print(self.input_qstn.text())
        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setAlignment(QtCore.Qt.AlignCenter)
        
        font=QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.l1.setFont(font)
        self.l1.setText("<font color='black'> Choose the image file </font>")
        self.lbl_qstn.setFont(font)
        self.lbl_qstn.setText("<font color='black'> Question </font>")
        self.lbl_output.setFont(font)
        self.lbl_output.setText("<font color='black'> Answer </font>")
        
        self.te = QtWidgets.QTextEdit()
        font1 = QtGui.QFont()
        font1.setFamily('Lucida')
        font1.setFixedPitch(True)
        font1.setPointSize(20)
        font1.setBold(True)
        self.te.setFont(font1)
        self.input_qstn.setFont(font1)
        
        self.img_input=QtWidgets.QLabel()
        self.img_input.resize(self.img_input.sizeHint())  
        self.img_input.setAlignment(QtCore.Qt.AlignCenter)
        
        self.img_output=QtWidgets.QLabel()
        self.img_output.setAlignment(QtCore.Qt.AlignCenter)
        self.img_output.resize(self.img_output.sizeHint())        
        
        
        self.btn_browse=QtWidgets.QPushButton("Browse")        
        self.btn_browse.clicked.connect(self.Browse)
        self.btn_browse.resize(self.btn_browse.sizeHint())

        self.btn_start=QtWidgets.QPushButton("PREDICT")        
        self.btn_start.clicked.connect(self.start_prediction)
        self.btn_start.resize(self.btn_start.sizeHint())  
        
        self.btn_close=QtWidgets.QPushButton("QUIT")        
#        self.btn_close.clicked.connect(self.close_event)
        self.btn_close.clicked.connect(self.close)
        self.btn_close.resize(self.btn_close.sizeHint())
        
        layout1 = QtWidgets.QHBoxLayout()
        layout1.addWidget(self.l1)
        layout1.addWidget(self.btn_browse)
        
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.lbl_qstn)
        layout2.addWidget(self.input_qstn)
          
        vbox_inpt=QtWidgets.QVBoxLayout()
        vbox_inpt.setContentsMargins(0,0,0,0)
        vbox_inpt.addLayout(layout1)
        vbox_inpt.addLayout(layout2)
#        vbox_inpt.addWidget(self.btn_browse)
        vbox_inpt.addWidget(self.img_input)
        
        vbox_opt=QtWidgets.QVBoxLayout()
        vbox_opt.setContentsMargins(0,0,0,0)
        vbox_opt.addWidget(self.lbl_output)
        vbox_opt.addWidget(self.progress)
        vbox_opt.addWidget(self.te)
        
#        hbox2.addStretch(0)   
        
        hbox=QtWidgets.QHBoxLayout()
        hbox.addLayout(vbox_inpt)
        hbox.addLayout(vbox_opt)
        
        vbox_main=QtWidgets.QVBoxLayout()
        vbox_main.addLayout(hbox)
#        vbox_main.addWidget(self.te)
        vbox_main.addWidget(self.btn_start)
        vbox_main.addWidget(self.btn_close)
#        fbox.addRow(hbox1)
        
        self.setLayout(vbox_main)
        self.setGeometry(200, 200, 1200, 700)
        self.setWindowTitle("VQA-DEMO-demo")
        self.setWindowIcon(QtGui.QIcon('vqa_logo.png'))

        self.fname=None
        self.result=None
        
#        self.progress.setGeometry(200, 80, 250, 20)
        self.show()     
       
    def Browse(self):

        w = QtWidgets.QWidget()            
        QtWidgets.QMessageBox.information(w,"Message", "Please select an image file")          
        
        filePath,_ = QtWidgets.QFileDialog.getOpenFileName(self, '*.')
        print('filePath',filePath, '\n')
        self.fname=str(filePath)
        self.img_input.setPixmap(QtGui.QPixmap(filePath))
         #                                     "/projectnb/ece601/MASS_VQA/VizWiz-VQA-PyTorch-master/test/VizWiz_test_00000176.jpg"
        self.img_input.setScaledContents(True)
        self.image_file_name=self.fname
        
        #self.grid = QtWidgets.QGridLayout()
        #self.grid.addWidget(self.img_input,1,1)
        #self.setLayout(self.grid)

        #self.setGeometry(50,50,320,200)
        #self.setWindowTitle("PyQT show image")
        self.show()
        
        
    def start_prediction(self):

        w = QtWidgets.QWidget()
        self.completed = 0
        self.te.setText('')
        parser = argparse.ArgumentParser()
        parser.add_argument('--path_config', default='config/default.yaml', type=str, help='path to a yaml config file')
        args = parser.parse_args()
        
        if args.path_config is not None:
            with open(args.path_config, 'r') as handle:
                config = yaml.load(handle)
        
        cudnn.benchmark = True # Generate dataset and loader
        print("Loading samples to predict from %s" % os.path.join(config['annotations']['dir'],config['prediction']['split'] + '.json'))

		# Load annotations
        path_annotations = os.path.join(config['annotations']['dir'], config['prediction']['split'] + '.json')
        input_annotations = json.load(open(path_annotations, 'r'))
        #print(input_annotations[1]['image'])
		
		# Data loader and dataset
        input_loader = vqa_dataset.get_loader(config, split=config['prediction']['split'])

		# Load model weights
        print("Loading Model from %s" % config['prediction']['model_path'])
        log = torch.load(config['prediction']['model_path'])

		# Num tokens seen during training
        num_tokens = len(log['vocabs']['question']) + 1
		# Use the same configuration used during training
        train_config = log['config']

        model = nn.DataParallel(models.Model(train_config, num_tokens)).cuda()

        dict_weights = log['weights']
        model.load_state_dict(dict_weights)
        predicted, samples_ids = predict_answers(model, input_loader, split=config['prediction']['split'])
        print(torch.FloatTensor(predicted))#, samples_ids)
        #print(input_annotations, predicted, samples_ids, input_loader.dataset.vocabs)
        submission = create_submission(input_annotations, predicted, samples_ids, input_loader.dataset.vocabs)

		#with open(config['prediction']['submission_file'], 'w') as fd:
		#    json.dump(submission, fd)
        print(submission)
        self.te.setText('Top Predictions is: ' + '\n' +  '\n'+submission)
        
        
        
        #self.result=[]
        #for label in reversed(y_sort_index[0,-5:]):
         #   print str(round(y_output[0,label]*100,2)).zfill(5)+ " % "+ labelencoder.inverse_transform(label)
         #   cmd=str(round(y_output[0,label]*100,2)).zfill(5)+ " % "+ labelencoder.inverse_transform(label)
#        #    stdouterr = os.popen4(cmd)[1].read()
         #   self.result.append(cmd)
	#self.te.setText('Top Predictions is: ' + '\n' +  '\n'+submission)





def main():
	
	app = QtWidgets.QApplication(sys.argv)
	ex = VQA_demo()
	app.exec_()
    


if __name__ == '__main__':
    main()
