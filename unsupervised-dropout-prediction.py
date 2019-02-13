#!/usr/bin/env python

import sys
import random
import numpy as np
import pandas as pd
import datetime
import json
import os
import pdfkit
import time
import os.path as op
import matplotlib.pyplot as plt
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders


class Net:
    
    def __init__(self, name, directed=False):
        self.directed=directed
        self.name=name
        self.nodes = {}
        self.userData = {}
        
    def addNode(self, nodeKey, node):
        if(nodeKey not in self.nodes):
            self.nodes[nodeKey]=node
        
    def addEdge(self, nodeKeySource, nodeKeyTarget, weight):
        
        if(self.directed==False):
            self.nodes[nodeKeySource].addOut(nodeKeyTarget, weight)
            self.nodes[nodeKeyTarget].addIn(nodeKeySource, weight)
            self.nodes[nodeKeySource].addIn(nodeKeyTarget, weight)
            self.nodes[nodeKeyTarget].addOut(nodeKeySource, weight)
        else:
            self.nodes[nodeKeySource].addOut(nodeKeyTarget, weight)
            self.nodes[nodeKeyTarget].addIn(nodeKeySource, weight)
        
    def printNodes(self):
        for nodeKey in self.nodes:
            print("Node: "+nodeKey)
            print("Labeled: "+str(self.nodes[nodeKey].isLabeled()))
            print("F: "+str(self.nodes[nodeKey].getF()))
            
            nodesOut = self.nodes[nodeKey].getOut()
            for node in nodesOut:
                print("\t->" + node + " ["+str(nodesOut[node])+"]")
                
            nodesIn = self.nodes[nodeKey].getIn()
            for node in nodesIn:
                print("\t<-" + node + " ["+str(nodesIn[node])+"]")

    
    def getNodes(self):
        return self.nodes
    
    def setUserData(self,key,value):
        self.userData[key]=value
        
    def getUserData(self,key):
        return self.userData[key]

class Node:
    
    def __init__(self, nodeName, nodeType):
        self.nodeName = nodeName
        self.nodeType = nodeType
        self.inNeighbors = {}
        self.outNeighbors = {}
        self.labeled = False
        self.f = None
        self.stats = {}

    def addIn(self, nodeKey, weight):
        self.inNeighbors[nodeKey]=weight
        
    def addOut(self, nodeKey, weight):
        self.outNeighbors[nodeKey]=weight
        
    def getIn(self):
        return self.inNeighbors
        
    def getOut(self):
        return self.outNeighbors
    
    def getInDegree(self):
        if ('inDegree' not in self.stats):
            self.stats['inDegree'] = float(len(self.getIn().keys()))
        return self.stats['inDegree']
        
    def getOutDegree(self):
        if ('outDegree' not in self.stats):
            self.stats['outDegree'] = float(len(self.getOut().keys()))
        return self.stats['outDegree']
    
    def getWeightedInDegree(self):
        if ('weightedInDegree' not in self.stats):
            self.stats['weightedInDegree'] = float(0)
            for nodeKey in self.getIn():
                self.stats['weightedInDegree'] += self.getIn()[nodeKey]
        return self.stats['weightedInDegree']
        
    def getWeightedOutDegree(self):
        if ('weightedOutDegree' not in self.stats):
            self.stats['weightedOutDegree'] = float(0)
            for nodeKey in self.getOut():
                self.stats['weightedOutDegree'] += self.getOut()[nodeKey]
        return self.stats['weightedOutDegree']
    
    def setLabeled(self,b):
        self.labeled=b

    def isLabeled(self):
        return self.labeled
        
    def setF(self,f):
        self.f=np.array(f)
    
    def getF(self):
        return self.f

    

def walking(graph, regularizer, max=1000, convergence=0.001):
    
    nodes = list(graph.getNodes().keys())
    iteration = 1
    print('Starting...')

    while(True):
        
        loss = 0
        random.shuffle(nodes)
        
        # student layer
        for nodeKey in nodes:
            if("student" in nodeKey):
                loss += regularizer.propagateAction2Student(graph,nodeKey)
        
        # action layer
        for nodeKey in nodes:
            if("action" in nodeKey):
                loss += regularizer.propagateStudent2Action(graph,nodeKey)

        print('Iteration '+str(iteration)+" | Loss="+str(loss))
        iteration+=1
        if(iteration >= max): break
        if(loss < convergence): break
    print("It's done.") 

class GraphRegularizationDropoutProfile:
    
    def __init__(self, maxPathSize=7):
        self.maxPathSize = maxPathSize

    def propagateAction2Student(self,graph,nodeKey):
        nodes = graph.getNodes()
        f_student = nodes[nodeKey].getF()
        loss = 0
        
        #print("node = "+nodeKey)
        #print("f = "+str(f_student))
        
        max_student_action_weight = 0
        for nodeIn in nodes[nodeKey].getIn():
            student_action_weight = nodes[nodeIn].getWeightedInDegree()
            if(student_action_weight > max_student_action_weight): max_student_action_weight = student_action_weight
            
        f_new = np.array([0]*len(f_student))
        for nodeIn in nodes[nodeKey].getIn():
            f_action = nodes[nodeIn].getF()
            student_action_weight = nodes[nodeKey].getIn()[nodeIn]
            student_action_weight_norm = student_action_weight / max_student_action_weight
            f_new = f_new + f_action*student_action_weight_norm

        loss = np.sum(np.abs(f_student - f_new))
        nodes[nodeKey].setF(f_new)
            
        #print("f_new = "+str(f_new))        
        
        return loss

    def propagateStudent2Action(self,graph,nodeKey):
        nodes = graph.getNodes()
        f_action = nodes[nodeKey].getF()
        loss = 0
        
        #print("node = "+nodeKey)
        #print("f = "+str(f_action))
        
        max_student_degree = 0
        max_student_action_weight = 0
        for nodeIn in nodes[nodeKey].getIn():
            student_degree = nodes[nodeIn].getOutDegree()
            student_action_weight = nodes[nodeKey].getIn()[nodeIn]
            if(student_degree > max_student_degree): max_student_degree = student_degree
            if(student_action_weight > max_student_action_weight): max_student_action_weight = student_action_weight
            
        f_new = np.array([0]*len(f_action))
        for nodeIn in nodes[nodeKey].getIn():
            f_student = nodes[nodeIn].getF()
                
            student_degree = nodes[nodeIn].getOutDegree()
            student_degree_norm = student_degree/max_student_degree
                
            student_action_weight = nodes[nodeKey].getIn()[nodeIn]
            student_action_weight_norm = student_action_weight / max_student_action_weight
                
            f_new = f_new + f_action*0.9 + ((student_degree_norm*student_action_weight_norm)*f_student)*0.1
        
        f_new = (f_new/(nodes[nodeKey].getInDegree()))
            
        loss = np.sum(np.abs(f_action - f_new))
        nodes[nodeKey].setF(f_new)
            
        #print("f_new = "+str(f_new))        
        
        return loss

    

        
        
    def jump(self,graph,nodeKey):
        nextNodes = list(graph.getNodes()[nodeKey].getOut().keys())
        if(len(nextNodes) > 0) :
            random.shuffle(nextNodes)
            return nextNodes[0]
        else: return None

        
def load(csv_data):
    
    courses = {}
    graphs = {}
    
    
    index=1;
    with open(csv_data) as fileobj:
        for line in fileobj:
            if(len(line.strip())==0): continue
            data = line.strip().split(';')
            if(len(data)!=5):
                print("Error in data input. Line "+str(index)+" ["+line.strip()+"]")
                sys.exit(-1)
            index+=1
            
            course = data[3]
            if(course not in courses): courses[course]={}
            
            student = data[2]
            if(student not in courses[course]): courses[course][student]={}
                
            action = data[4]
            if(action not in courses[course][student]): courses[course][student][action]=1
            else: courses[course][student][action]+=1
    
    
    for course in courses:
        graph = Net(course,False)
        actions = {}
        
        for student in courses[course]:
            for action in courses[course][student]:
                weight = courses[course][student][action]                
                node1 = Node(student, 'student')
                node1key = student+":student"
                node2 = Node(action, 'action')
                node2key = action+":action"
                graph.addNode(node1key,node1)
                graph.addNode(node2key,node2)
                graph.addEdge(node1key,node2key,weight)
                actions[node2key]=1
        
        
        graph.setUserData('course_actions',list(actions.keys()))
        course_actions = graph.getUserData('course_actions')
        
        # data labeling        
        for node in graph.getNodes():
            if(node in course_actions):
                f = [0]*len(course_actions)
                i = course_actions.index(node)
                f[i]=1
                f = np.array(f)
                graph.getNodes()[node].setLabeled(True)
                graph.getNodes()[node].setF(f)
            else:
                f = [0]*len(course_actions)
                f = np.array(f)
                graph.getNodes()[node].setF(f)
        
        graphs[course] = graph
                
            
    return graphs


def dropout_risk(graphs, minStudents):
    
    output = {}
    
    for course in graphs:
        
        print('Course='+course)
        output[course] = {}
        output[course]['actions']=graphs[course].getUserData('course_actions')
        
        nodes = graphs[course].getNodes()
        
        students = {}
        
        output[course]['F'] = {}

        f_total = np.array([0]*len(output[course]['actions']))
        
        df = pd.DataFrame(columns=['risk_factor'])
        for node in nodes:
            if(":student" in node):
                risk_factor = np.sum(nodes[node].getF())
                students[node] = risk_factor
                df = df.append({'risk_factor': risk_factor}, ignore_index=True)
                if node not in output[course]['F']:
                    
                    f_total = f_total + nodes[node].getF()
                    
                    output[course]['F'][node]=list(nodes[node].getF())
        
        
        output[course]['f_total']=list(f_total)
        print("f_total"+str(f_total))
        output[course]['students']=students

        quantiles = df.risk_factor.quantile([0.05,0.1,0.15])
        output[course]['thresholds']=list(quantiles)
        
        output[course]['risk']={}
        output[course]['risk']['critical']=[]
        output[course]['risk']['high']=[]
        output[course]['risk']['medium']=[]
        output[course]['risk']['low']=[]
        
        if(len(students) < minStudents): continue
    
        f_low = np.array([0]*len(output[course]['actions']))
    
        rank = sorted(students.items(), key=lambda kv: kv[1])
        for student in rank:
            if(student[1] < quantiles[0.05]):
                output[course]['risk']['critical'].append(student[0])
            if(student[1] >= quantiles[0.05] and student[1] < quantiles[0.1]):
                output[course]['risk']['high'].append(student[0])
            if(student[1] >= quantiles[0.1] and student[1] < quantiles[0.15]):
                output[course]['risk']['medium'].append(student[0])
            if(student[1] >= quantiles[0.15]):
                output[course]['risk']['low'].append(student[0])
                f_low = f_low + nodes[student[0]].getF()
        
        
        output[course]['f_low']=list(f_low)
        
        output[course]['prior'] = {}
        
        for node in nodes:
            if(":student" in node):
                c = np.corrcoef(f_low, nodes[node].getF())[0, 1]
                if(c < 0): c = 0
                priority = 1.0 - c
                output[course]['prior'][node]=priority

                
    return output
        
    
coursesName = {}
def readCourses(csv_data):
    index=0
    with open(csv_data) as fileobj:
        for line in fileobj:
            if(len(line.strip())==0): continue
            data = line.strip().split(';')
            if(len(data)!=2):
                print("Error in data input. Line "+str(index)+" ["+line.strip()+"]")
                sys.exit(-1)
            index+=1
            coursesName[data[0]]=data[1]


def getCourseName(course):
    s = course.split(":")[0]
    if( s in coursesName): return coursesName[s]
    return 'Course '+s


studentsName = {}
def readStudents(csv_data):
    index=0
    with open(csv_data) as fileobj:
        for line in fileobj:
            if(len(line.strip())==0): continue
            data = line.strip().split(';')
            if(len(data)!=2):
                print("Error in data input. Line "+str(index)+" ["+line.strip()+"]")
                sys.exit(-1)
            index+=1
            studentsName[data[0]]=data[1]

def getStudentName(student):
    s = student.split(":")[0]
    if( s in studentsName): return studentsName[s]
    return 'Student '+s


def getReport(dropout):

    
    if os.path.exists('report.html'):
        os.remove('report.html')
        

    f1=open('./report.html', 'a')
        
    types = ['critical','high', 'medium', 'low']


    s = "<html><head>  <title>Websensors</title>  <meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\">   <style type=\"text/css\">   body {font-family: \"Trebuchet MS\", Arial, sans-serif;}  </style> </head><table border=0 cellpadding=\"20\" cellspacing=\"0\" bgcolor=\"#273140\" width=\"100%\"><tr><td align=\"center\"><font color=\"white\" style=\"font-family: 'Russo One', sans-serif; font-size: 220%;\">Websensors - School Dropout Prediction (Report)</font></td></tr></table>\n"
    f1.write(s)    

    s = "<br><br><center><b>Date: "+str(datetime.date.today())+"</b></center>\n"
    f1.write(s)
    
    s = "<br><br><center><img src='pie.png'></center><br><br>\n"
    f1.write(s)
    
    s = "<table border='2' cellpadding='2' cellspacing='2' width=\"100%\"><tr><td width=\"45%\"><b>Student</b></td><td width=\"45%\"><b>Course</b></td><td width=\"10%\"><b>Dropout Risk</b></td></tr>\n"
    f1.write(s)
    
    for course in dropout:

        if('risk' in dropout[course]):
            for t in types:
                for student in dropout[course]['risk'][t]:
                    risk_factor = dropout[course]['students'][student]
                    priority = dropout[course]['prior'][student]
                    s = "<tr><td>"+getStudentName(student)+"</td>"
                    s += "<td>"+getCourseName(course)+"</td>"
                    s += "<td>"+t+"</td></tr>\n"
                    f1.write(s)
    

    total_critical = 0
    total_high = 0
    total_medium = 0
    total_low = 0
    for course in dropout:
        if('risk' in dropout[course]):
            total_critical += len(dropout[course]['risk']['critical'])
            total_high += len(dropout[course]['risk']['high'])
            total_medium += len(dropout[course]['risk']['medium'])
            total_low += len(dropout[course]['risk']['low'])

        
    s = "</table><hr>\n"
    f1.write(s)
    s = "<table border=0 cellpadding=\"20\" cellspacing=\"0\" bgcolor=\"#273140\" width=\"100%\"><tr><td align=\"center\"><font color=\"white\" style=\"font-family: 'Russo One', sans-serif; font-size: 80%;\">More details: <a href=\"https://websensors.net.br/ead/\">https://websensors.net.br/ead/</a></font></td></tr></table></body></html>"
    f1.write(s)
    
    # Data to plot
    labels = 'Critical', 'High', 'Medium', 'Low'
    sizes = [total_critical, total_high, total_medium, total_low]
    colors = ['red', 'yellow', 'green', 'gray']
    explode = (0.2, 0, 0, 0)  # explode 1st slice

    # Plot
    fig = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    plt.title('School Dropout Risk')
    plt.savefig('pie.png')   # save the figure to file
    
    
    
def sendMail(mail_from,mail_to,mail_smtp,mail_password,mail_message):
    # create message object instance
    msg = MIMEMultipart()


    # setup the parameters of the message
    password = mail_password
    msg['From'] = mail_from
    msg['To'] = mail_to
    msg['Subject'] = "[Websensors-EAD] School Dropout Prediction Report"
    
    # msg
    msg.attach(MIMEText(mail_message))

    # attach pdf
    path='report.pdf'
    part = MIMEBase('application', "octet-stream")
    with open(path, 'rb') as file:
        part.set_payload(file.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition',
                        'attachment; filename="{}"'.format(op.basename(path)))
    msg.attach(part)
    

    # create server
    server = smtplib.SMTP(mail_smtp)

    server.starttls()

    # Login Credentials for sending the mail
    server.login(msg['From'], password)


    # send the message via the server.
    server.sendmail(msg['From'], msg['To'], msg.as_string())

    server.quit()

    print("Sending mail: OK")



    
def websensors(config):

    graphs = load(config['clickstream'])
    readStudents(config['students'])
    readCourses(config['courses'])

    for course in graphs:
        regularizer = GraphRegularizationDropoutProfile(maxPathSize=config['net_regularizer_path_size'])
        walking(graphs[course],regularizer,max=config['net_regularizer_max_iterations'])    

    res = dropout_risk(graphs,config['min_students_by_course'])
    getReport(res)

    pdfkit.from_file('report.html', 'report.pdf')

    if(config['mail_report']==True):
        sendMail(config['mail_from'],config['mail_to'],config['mail_smtp'],config['mail_pass'],config['mail_text'])

#################################################################
#################################################################
#################################################################
#################################################################
#################################################################
######################## CONFIGURATION ##########################
#################################################################
config={}
# student patterns file
config['clickstream']='data/students-patterns.csv'
config['courses']='data/courses.csv'
config['students']='data/students.csv'

# machine learning configuration
config['net_regularizer_path_size']=100
config['net_regularizer_max_iterations']=100
config['min_students_by_course']=10

# notification parameters
config['mail_report']=False
config['mail_from']='abc@gmail.com'
config['mail_to']='123@gmail.com'
config['mail_smtp']='smtp.gmail.com: 587'
config['mail_pass']='12345'
config['mail_text']="Dear administrator,\n\nI'm attaching the School Dropout Prediction Report.\n\nI suggest the application of mitigation policies for students classified in the 'critical' and 'high' risk group.\n\nRegards,\nWebsensors-EAD AI Bot.\n[I'm a bot. Do not answer this message.]"


#################################################################
#################################################################

# Running

websensors(config)
