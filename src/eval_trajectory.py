import openai
import os
import json
import copy
import sys
import os
import pdb
import re
from collections import namedtuple
import numpy as np
import re

openai.api_key="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"


def matching_score_prompt(traj_info:dict,description):
    """
    instructional
    """

    #note that the step-by-step (i.e. chain of thought) reasoning is critical for the coherence of the score. 
    prompt="""Let us consider a square of size 200x200. The origin is fixed at the bottom left corner, and the x,y axes are respectively horizontal and vertical (with x looking towards the right, i.e. east, and y looking upwards, i.e. north). Let us define a python dictionary to represent point that have been sampled from a 2d trajectory in that square. The dictionary will have the following keys: dict_keys(['timestep', 'pos', 'semantics', 'colors']). Here is the explanation for each: 1) The complete trajectories are composed of N points, and each has a timestep t_i (with i ranging from 0 to N-1).  The 'timestep' key corresponds to the t_i of the sampled point. 2) the 'pos' key is for the 2d position of the point, expressed as (x,y) in the coordinate frame defined above. 3) The square actually represents a room, where there are several objects such as a fridge, a chair, a cactus and so on. The 'semantics' gives information on objects to which the point are close (name of objects, and where the agent is situated w.r.t those objects, e.g. to the east or north of the cactus, etc). 4) The room which is represented by the 200x200 square also has tiles of different colors in different areas. The 'colors' key gives information about the tile color where the 2d point is. You will receive a trajectory dict afte the tag [TRAJ], and its its natural language description will be given after the [NAT] tag. Your task is evaluate how well the trajectory dict and its description match. You must write a short textual evaluation of that similarity after the tag. [REASON] You MUST pay attention to the order in which the objects are visited. Similarly, the order in which the tiles appear is important. After you have written the reason, you will also output at numerical score in [0,1] that you will prefix with 'score=='."""# before giving a score in [0,1] after the given tag [SCR]."""

    prompt=prompt+"\n"+f"[TRAJ] {traj_info} \n [NAT] {description} \n \n [REAS] \n"


    return prompt


def round_pos_info(traj_lst,round_n=1):

    for s_i in range(len(traj_lst)):
        for ii in range(2):
            traj_lst[s_i]["pos"][ii]=round(traj_lst[s_i]["pos"][ii],round_n)

    return traj_lst


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


if __name__=="__main__":


    if len(sys.argv)!=3:
        print(f"usage: {sys.argv[0]} <input_file> <output_dir>. The input file should be a json containing a list with pairs (natural_language_descr, traj_lst)")
    
    with open(sys.argv[1],"r") as fl:
            pair_lst=json.load(fl)
    
    natural_language_descr_lst=[x[0] for x in pair_lst]
    #traj_descr_lst=round_pos_info(dumb_down_traj_for_gpt3([x[1] for x in pair_lst]))
    traj_descr_lst=[round_pos_info(dumb_down_traj_for_gpt3(x[1])) for x in pair_lst]

    result_dict={"msg":[], "score":[]}
    counter=0
    for nld, td_lst in zip(natural_language_descr_lst, traj_descr_lst):

        print("counter==",counter)
      
        prompt=matching_score_prompt(td_lst, nld)


        response=openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{prompt}"}],
                temperature=0.05)

        msg=response["choices"][0]["message"]["content"]
        match_re = re.search(r'score==([0-9\.]+)', msg)
        score=float(match_re.group(1))

        result_dict["msg"].append(msg)
        result_dict["score"].append(score)

        counter+=1
    with open(sys.argv[2]+"/scoring.json","w") as fl:
        json.dump(result_dict,fl)


    


