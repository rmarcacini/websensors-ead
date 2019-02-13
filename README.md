# [Websensors-EAD] School Dropout Prediction Report

**Websensors-EAD** tool analyzes student access data in online distance learning environments such as Moodle, MOOCS, etc. The goal is to identify **dropout patterns** and organize students into four dropout risk groups: critical, high, medium, and low. Websensors-EAD uses regularization in heterogeneous networks [1] as a machine learning method to analyze school dropout patterns.

Websensors-EAD is part of the [Websensors Analytics](https://websensors.net.br) research project, developed at the [Laboratory of Scientific Computing (Lives)](http://lives.ufms.br) of the Federal University of Mato Grosso do Sul (UFMS), Três Lagoas, Brazil.

The Websensors-EAD source code is free for academic purposes. For private purposes, contact the project team.


## Data preparation

The data input of the Websensors-EAD is the students access log of the online learning environment system. The data should be tabulated in a CSV format, using the character ";" as a delimiter. The data must have five columns: access identifier (unique), date or timestamp, student id, course id, and resource name.

The data directory contains an example of an input file.

In practice, Websensors-EAD analyzes input patterns as a clickstream graph. An example of a clickstream graph is shown in the figure below.

![Example of a Clickstream Graph (From [1])](https://i.imgur.com/HnJkMge.png)

Example of a Clickstream Graph (From [2])

## Configuration

Edit the **unsupervised-dropout-prediction.py** file and change the following lines.

> \# **student patterns file**

> config['clickstream']='data/students-patterns.csv'

> config['courses']='data/courses.csv'

> config['students']='data/students.csv'

> \# **machine learning configuration**

> config['net_regularizer_path_size']=100

> config['net_regularizer_max_iterations']=100

> config['min_students_by_course']=10

If you want Websensors-EAD to send a PDF report (via e-mail) to students who are at risk of dropout, change the config['mail_report'] setting to True and informs your smtp configuration.

> \# **notification parameters**

> config['mail_report']=False

> config['mail_from']='abc@gmail.com'

> config['mail_to']='123@gmail.com'

> config['mail_smtp']='smtp.gmail.com: 587'

> config['mail_pass']='12345'

> config['mail_text']="Dear administrator,\n\nI'm attaching the School Dropout Prediction Report.\n\nI suggest the application of mitigation policies for students classified in the 'critical' and 'high' risk group.\n\nRegards,\nWebsensors-EAD AI Bot.\n[I'm a bot. Do not answer this message.]"


## Running Websensors-EAD

To run Websensors-EAD, install the numpy, pandas and matplotlib libraries for Python 3.

* pip install numpy
* pip install pandas
* pip install matplotlib

Run the **unsupervised-dropout-prediction.py** script:

* python unsupervised-dropout-prediction.py

It is recommended to schedule periodic execution  (e.g. weekly) of this script, for example, using the Cron tool in Linux environments.

Websensors-EAD generates a PDF report organizing students into four dropout risk groups. An example report is available in the **report.pdf** file.

## References

[1] SHI, Chuan et al. A survey of heterogeneous information network analysis. **IEEE Transactions on Knowledge and Data Engineering**, v. 29, n. 1, p. 17-37, 2017.

[2] GEIGLE, Chase; ZHAI, ChengXiang. Modeling mooc student behavior with two-layer hidden markov models. In: **Proceedings of the Fourth (2017) ACM Conference on Learning@ Scale**. ACM, 2017. p. 205-208.
