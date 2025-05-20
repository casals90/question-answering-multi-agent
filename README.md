## Project
TODO

<div align="center">
  <img src="notebooks/images/multi_agents_graph.png" alt="Multi-Agent Pipeline Graph"/>
  <p><em>Figure 1: Multi-Agent Pipeline Graph showing different processing routes</em></p>
</div>

Project Organization
------------
    ├── LICENSE
    ├── Makefile             
    ├── README.md             
    ├── config
    │   ├── requirements.txt  
    │   ├── settings.yaml     
    ├── data
    │   ├── external         
    │   ├── interim
    │   ├── processed        
    │   └── raw
    ├── notebooks
    │   ├── answer-gaia-questions.ipynb        
    │   └── answer-simple-question.ipynb
    │   └── images
    │      └── multi_agents_graph.png
    ├── src
    │   └── agent  
    │      └── prompt.py
    │      └── question_answering.py
    │      └── tool.py
    │      └── utils.py
    │      └── workflow.py
    │   └── data
    │      └── extract.py
    │      └── load.py
    │   └── tools 
    │      └── audio.py
    │      └── startup.py
    │      └── utils.py

--------

How to run the code
------------

### Create environment file

Rename the '.env.demo' as '.env' and fill in.

This command shows the user ID and group ID. 
```commandline
printf "UID=$(id -u)\nGID=$(id -g)\n"
```

### Check the volumes

The system needs to interact with some folders in order to be able to take
necessary data, train models or predict. From the root folder, the volumes 
are mounted as follows:

```commandline
volumes:
  - ./data/models:/data/models
  - ./data/processed:/data/processed
  - ./data/raw:/data/raw
  - ./reports:/reports
```

Hence, the volumes are located inside the data folder.
Should the above locations be inconvenient, feel free to change
the 'docker-compose.yml' file to change them.
E.g. /myownpath/some_raw_data:/data/raw

### Utility to start/shutdown the system

To facilitate system startup and shutdown, there is a script available.
Please execute the following command from the root folder:

```commandline
make deploy_jupyter
```

This command initiates the Jupyterlab environment and displays the logs in
the command line. To shut down the system, press "CTRL + C".

Shutdown the system as follows:

```commandline
make jupyter_down
```


