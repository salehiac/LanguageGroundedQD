import openai
import os
import json
import copy
import sys

openai.api_key="INSERT KEY HERE"


def generate_description(traj_info:dict):

        prompt="""Let us consider a square of size 200x200. The origin is fixed at the bottom left corner, and the x,y axes are respectively horizontal and vertical (with x looking towards the right, and y looking upwards). Let us define a python dictionary to represent point that have been sampled from a 2d trajectory in that square. The dictionary will have the following keys: dict_keys(['timestep', 'pos', 'semantics', 'colors']). Here is the explanation for each: 1) The complete trajectories are composed of N points, and each has a timestep t_i (with i ranging from 0 to N-1).  The 'timestep' key corresponds to the t_i of the sampled point. 2) the 'pos' key is for the 2d position of the point, expressed as (x,y) in the coordinate frame defined above. 3) The square actually represents a room, where there are several objects such as a fridge, a chair, a cactus and so on. The 'semantics' gives information on objects to which the point are close (name of objects). 4) The room which is represented by the 200x200 square also has tiles of different colors in different areas. The 'colors' key gives information about the tile color where the 2d point is. Your task is to describe such trajectories with text, without any numerical values. 

[TRAJECTORY] [{'timestep': 0, 'pos': [20.818174997965492, 49.23204549153647], 'semantics': ['to the north  of fridge'], 'colors': 'pink'}, {'timestep': 40, 'pos': [56.35655721028646, 58.72019449869791], 'semantics': [], 'colors': 'pink'}, {'timestep': 80, 'pos': [91.18633015950522, 75.68450927734376], 'semantics': [], 'colors': 'orange'}, {'timestep': 120, 'pos': [97.47745768229167, 118.49943033854167], 'semantics': ['to the north west of statue'], 'colors': 'orange'}, {'timestep': 160, 'pos': [65.02020772298177, 146.15777587890625], 'semantics': ['to the north  of fan'], 'colors': 'red'}, {'timestep': 200, 'pos': [33.05481719970703, 160.7065912882487], 'semantics': ['to the south east of cabinet'], 'colors': 'red'}, {'timestep': 240, 'pos': [25.741124471028648, 121.62562052408855], 'semantics': [], 'colors': 'yellow'}, {'timestep': 280, 'pos': [60.53137207031249, 101.75295003255208], 'semantics': ['on bathtub'], 'colors': 'yellow'}, {'timestep': 320, 'pos': [93.50303141276042, 114.1073710123698], 'semantics': ['to the north west of statue'], 'colors': 'orange'}, {'timestep': 360, 'pos': [65.66576131184895, 142.8917999267578], 'semantics': ['on fan'], 'colors': 'red'}, {'timestep': 399, 'pos': [35.575538635253906, 163.30083719889322], 'semantics': ['to the south east of cabinet'], 'colors': 'red'}]

[DESCR] Start near the fridge on a pink tile, then head right into an area with no nearby objects. Move towards the orange tiles, passing by the statue,  until you reach the fan. Walk towards the cabinet on the red tiles, then descend to the yellow tiles near the bathtub. Finally, circle back towards the statue and  return to the cabinet on red tiles."""

        prompt=prompt+"\n"+f"[TRAJECTORY] {traj_info} \n [DESCR]"


        return prompt


def dumb_down_traj_for_gpt3(traj_dict:dict):
    """
    gpt3.x gets confused by extra info about distances from other objects/tiles etc
    """
    ann=copy.deepcopy(traj_dict)
    for ii in range(len(ann)):
        ann[ii]["semantics"]=list(ann[ii]["semantics"].keys())
        ann[ii]["colors"]=ann[ii]["colors"][0]

    return ann



if __name__=="__main__":


    with open(sys.argv[1],"r") as fl:
        _traj_dict=json.load(fl)
   
    _traj_dict=dumb_down_traj_for_gpt3(_traj_dict)
    _prompt=generate_description(traj_info=_traj_dict)
    print("PROMPT:\n",_prompt)

    if 1:
        response = openai.Completion.create(
                model="text-davinci-003",
                #prompt=generate_description(traj_info=_traj_dict),
                prompt=_prompt,
                max_tokens=1000,
                temperature=0.6)

        print("RESPONSE\n",response["choices"][0]["text"])






