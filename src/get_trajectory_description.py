import openai
import os
import json
import copy
import sys
import os
import pdb
import re
from collections import namedtuple

openai.api_key="ADD YOUR KEY HERE"

def generate_description_narrative(traj_info:dict):

        prompt="""Let us consider a square of size 200x200. The origin is fixed at the bottom left corner, and the x,y axes are respectively horizontal and vertical (with x looking towards the right, i.e. east, and y looking upwards, i.e. north). Let us define a python dictionary to represent point that have been sampled from a 2d trajectory in that square. The dictionary will have the following keys: dict_keys(['timestep', 'pos', 'semantics', 'colors']). Here is the explanation for each: 1) The complete trajectories are composed of N points, and each has a timestep t_i (with i ranging from 0 to N-1).  The 'timestep' key corresponds to the t_i of the sampled point. 2) the 'pos' key is for the 2d position of the point, expressed as (x,y) in the coordinate frame defined above. 3) The square actually represents a room, where there are several objects such as a fridge, a chair, a cactus and so on. The 'semantics' gives information on objects to which the point are close (name of objects, and where the agent is situated w.r.t those objects, e.g. to the east or north of the cactus, etc). 4) The room which is represented by the 200x200 square also has tiles of different colors in different areas. The 'colors' key gives information about the tile color where the 2d point is. Your task is to describe such trajectories with text, without any numerical values. Note that if there is no significant motion during the entire trajectory, it is acceptable to give a very concise description such as 'stay near the starting point'. The trajectory will be given after the tag [TRAJECTORY], and I want you to write the description after [DESCR]."""

        prompt=prompt+"\n"+f"[TRAJECTORY] {traj_info} \n\n [DESCR]"


        return prompt

def generate_description(traj_info:dict):
    """
    instructional
    """

    prompt="""Let us consider a square of size 200x200. The origin is fixed at the bottom left corner, and the x,y axes are respectively horizontal and vertical (with x looking towards the right, i.e. east, and y looking upwards, i.e. north). Let us define a python dictionary to represent point that have been sampled from a 2d trajectory in that square. The dictionary will have the following keys: dict_keys(['timestep', 'pos', 'semantics', 'colors']). Here is the explanation for each: 1) The complete trajectories are composed of N points, and each has a timestep t_i (with i ranging from 0 to N-1).  The 'timestep' key corresponds to the t_i of the sampled point. 2) the 'pos' key is for the 2d position of the point, expressed as (x,y) in the coordinate frame defined above. 3) The square actually represents a room, where there are several objects such as a fridge, a chair, a cactus and so on. The 'semantics' gives information on objects to which the point are close (name of objects, and where the agent is situated w.r.t those objects, e.g. to the east or north of the cactus, etc). 4) The room which is represented by the 200x200 square also has tiles of different colors in different areas. The 'colors' key gives information about the tile color where the 2d point is. Your task is to describe such trajectories with text, without any numerical values. Note that if there is no significant motion during the entire trajectory, it is acceptable to give a very concise description such as 'stay near the starting point'. Also, try to make those descriptions instructional, as if you were trying to guide an agent. Important: please be concise. The trajectory will be given after the tag [TRAJECTORY], and I want you to write the description after [DESCR]."""

    prompt=prompt+"\n"+f"[TRAJECTORY] {traj_info} \n\n [DESCR]"


    return prompt




def generate_description_few_shot(traj_info:dict):

    """
    deprecating that as it seems that zero-shot works as well if not better
    """

    prompt="""Let us consider a square of size 200x200. The origin is fixed at the bottom left corner, and the x,y axes are respectively horizontal and vertical (with x looking towards the right, and y looking upwards). Let us define a python dictionary to represent point that have been sampled from a 2d trajectory in that square. The dictionary will have the following keys: dict_keys(['timestep', 'pos', 'semantics', 'colors']). Here is the explanation for each: 1) The complete trajectories are composed of N points, and each has a timestep t_i (with i ranging from 0 to N-1).  The 'timestep' key corresponds to the t_i of the sampled point. 2) the 'pos' key is for the 2d position of the point, expressed as (x,y) in the coordinate frame defined above. 3) The square actually represents a room, where there are several objects such as a fridge, a chair, a cactus and so on. The 'semantics' gives information on objects to which the point are close (name of objects). 4) The room which is represented by the 200x200 square also has tiles of different colors in different areas. The 'colors' key gives information about the tile color where the 2d point is. Your task is to describe such trajectories with text, without any numerical values. Note that if there is no significant motion during the entire trajectory, it is acceptable to give a very concise description such as 'stay near the starting point'. 

[TRAJECTORY] [{'timestep': 0, 'pos': [20.8, 49.2], 'semantics': ['to the north  of fridge'], 'colors': 'pink'}, {'timestep': 40, 'pos': [56.3, 58.7], 'semantics': [], 'colors': 'pink'}, {'timestep': 80, 'pos': [91.1, 75.6], 'semantics': [], 'colors': 'orange'}, {'timestep': 120, 'pos': [97.4, 118.4], 'semantics': ['to the north west of statue'], 'colors': 'orange'}, {'timestep': 160, 'pos': [65.1, 146.1], 'semantics': ['to the north  of fan'], 'colors': 'red'}, {'timestep': 200, 'pos': [33.1, 160.7], 'semantics': ['to the south east of cabinet'], 'colors': 'red'}, {'timestep': 240, 'pos': [25.7, 121.6], 'semantics': [], 'colors': 'yellow'}, {'timestep': 280, 'pos': [60.5, 101.7], 'semantics': ['on bathtub'], 'colors': 'yellow'}, {'timestep': 320, 'pos': [93.5, 114.1], 'semantics': ['to the north west of statue'], 'colors': 'orange'}, {'timestep': 360, 'pos': [65.6, 142.8], 'semantics': ['on fan'], 'colors': 'red'}, {'timestep': 399, 'pos': [35.5, 163.3], 'semantics': ['to the south east of cabinet'], 'colors': 'red'}]

[DESCR] Start near the fridge on a pink tile, then head right into an area with no nearby objects. Move towards the orange tiles, passing by the statue,  until you reach the fan. Walk towards the cabinet on the red tiles, then descend to the yellow tiles near the bathtub. Finally, circle back towards the statue and  return to the cabinet on red tiles."""

    prompt=prompt+"\n"+f"[TRAJECTORY] {traj_info} \n\n [DESCR]"


    return prompt

def round_pos_info(traj_lst,round_n=1):

    for s_i in range(len(traj_lst)):
        for ii in range(2):
            traj_lst[s_i]["pos"][ii]=round(traj_lst[s_i]["pos"][ii],round_n)


def dumb_down_traj_for_gpt3(traj_lst:list):
    """
    gpt3.x gets confused by extra info about distances from other objects/tiles etc
    """
    ann=copy.deepcopy(traj_lst)
    for ii in range(len(ann)):
        ann[ii]["semantics"]=list(ann[ii]["semantics"].keys())
        ann[ii]["colors"]=ann[ii]["colors"][0]

    return ann


def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None


InOutFilePair=namedtuple("InOutFilePair",["in_fn","out_fn"])

def fetch_description(in_dir,outdir,start_idx,end_idx,gpt3_mode=True):
    """
    fetches descriptions for files with index in the interval [start_idx, end_idx) 
    """

    fns={get_trailing_number(x[:-5]):InOutFilePair(in_fn=in_dir+"/"+x, out_fn=outdir+"/"+x) for x in os.listdir(in_dir)}
    fns_sorted=[fns[ii] for ii in range(len(fns))] 


    for ii in range(start_idx,end_idx):

        print(f"fetching description for trajectory {ii}...")

        with open(fns_sorted[ii].in_fn,"r") as fl:
            traj_lst=json.load(fl)

        traj_lst=dumb_down_traj_for_gpt3(traj_lst) if gpt3_mode else traj_lst 
        round_pos_info(traj_lst)
        prompt=generate_description(traj_info=traj_lst)

        #print(prompt)

        response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=1000,
                temperature=0.6)

        with open(fns_sorted[ii].out_fn,"w") as fl:
            resp_d={"descr":response["choices"][0]["text"]}
            json.dump(resp_d,fl)
    
    return True

if __name__=="__main__":


    test_get_single_traj_descr=False
    #test_get_single_traj_descr=True

    test_read_annotations=True
    #test_read_annotations=False


    if test_get_single_traj_descr:
        with open(sys.argv[1],"r") as fl:
            _traj_lst=json.load(fl)
   
        _traj_lst=dumb_down_traj_for_gpt3(_traj_lst)
        #_traj_lst_before=copy.deepcopy(_traj_lst)
        round_pos_info(_traj_lst)
        _prompt=generate_description(traj_info=_traj_lst)

        print("PROMPT:\n",_prompt)

        response = openai.Completion.create(
                model="text-davinci-003",
                #prompt=generate_description(traj_info=_traj_lst),
                prompt=_prompt,
                max_tokens=1000,
                temperature=0.6)

        print("RESPONSE\n",response["choices"][0]["text"])

    if test_read_annotations:

        _outdir=sys.argv[2]
        ###note: - up to 2890 (inclusive) have been generated with few-shot prompting (generate_description_few_shot function) and without rounding the pos
        ###      - from 2891 to 3000 have been generated with zero-shot (generate_description) and with rounding (so the price should be lower)
        ###        Interestingly, those examples result in narrative descriptions, while the previous one became instructinal (well, the example was instructional)
        ###        Interestingly, those examples result in narrative descriptions, while the previous one became instructinal (well, the example was instructional)
        ###      - from 3001 to 3027, they also are given the requirement to be instructional
        ###      - from 3028 to 3050, they also are given the requirement to be CONCISE
        _start_idx=4418#not annotated yet (exceeded quota...)
        _end_idx=6000
        
        fetch_description(sys.argv[1],outdir=_outdir,start_idx=_start_idx,end_idx=_end_idx)

