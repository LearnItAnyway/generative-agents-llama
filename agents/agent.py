import random
import networkx as nx
import torch
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_name_or_path = "/path/model_path"
model_basename = "model_basename"
#model_name_or_path = "Llama-2-70B-chat-GPTQ"
#model_basename = "gptq_model-4bit--1g"
#model_name_or_path = "Llama-2-13B-chat-GPTQ"
#model_basename = "gptq_model-4bit-128g"


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename,
        inject_fused_attention=False, # Required for Llama 2 70B model at this time.
        use_safetensors=True,
        trust_remote_code=False,
        device="cuda:0",
        use_triton=False,
        quantize_config=None)

#prompt = '''[INST] Who is the first person on the moon? [/INST]'''
def text_postprocess(text):
    # remove white spaces at the start and end of the text
    while text[0] in [' ', '\n', '\r']:
        text = text[1:]
    while text[-1] in [' ', '\n', '\r']:
        text = text[:-1]
    return text

def generate(prompt, temperature=0.7, max_length=128):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=temperature, max_new_tokens=max_length)
    out = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    if '[/INST]' in out:
        out = out.split('[/INST]')[1]
    if '[INST]' in out:
        out = out.split('[INST]')[0]
    out = text_postprocess(out)
    return out

def summarize_simulation(log_output):
    prompt = f"Summarize the simulation loop:\n\n{log_output}"
    response = generate(prompt)
    return response

def get_rating_fast(prompt):
    prompt_rate = f"[INST]{prompt} [/INST]I will go to "
    input_ids = tokenizer(prompt_rate, return_tensors='pt').input_ids.cuda()
    input_ids[0][-1] = 29871
    with torch.no_grad():
        output = model(input_ids)
    rate = output.logits[0, -1][[29896, 29906, 29941, 29946, 29945]].argmax().item()+1
    return rate

def get_best_location_fast(prompt):
    prompt_rate = f"[INST]{prompt} [/INST]Sure! Here's my rating:\n\nRating: "
    input_ids = tokenizer(prompt_rate, return_tensors='pt').input_ids.cuda()
    input_ids[0][-1] = 29871
    with torch.no_grad():
        output = model(input_ids)
    loc = output.logits[0, -1][[29900, 29896, 29906, 29941, 29946, 29945, 
                                29953, 29955, 29947, 29929, 29910]]
    return loc

class Agent:
     
    """
    A class to represent an individual agent in a simulation similar to The Sims.

    Attributes:
    -----------
    name : str
        The name of the agent.
    description : str
        A brief description of the agent.
    location : str
        The current location of the agent in the simulated environment.
    memories : list
        A list of memories the agent has about their interactions.
    compressed_memories : list
        A list of compressed memories that summarize the agent's experiences.
    plans : str
        The agent's daily plans, generated at the beginning of each day.

    Methods:
    --------
    plan(global_time, town_people, prompt_meta):
        Generates the agent's daily plan.
    
    execute_action(other_agents, location, global_time, town_areas, prompt_meta):
        Executes the agent's action based on their current situation and interactions with other agents.
    
    update_memories(other_agents, global_time, action_results):
        Updates the agent's memories based on their interactions with other agents.
    
    compress_memories(memory_ratings, global_time, MEMORY_LIMIT=10):
        Compresses the agent's memories to a more manageable and relevant set.
    
    rate_locations(locations, town_areas, global_time, prompt_meta):
        Rates different locations in the simulated environment based on the agent's preferences and experiences.
    """
     
    def __init__(self, name, description, starting_location, world_graph):
        self.name = name
        self.description = description
        self.location = starting_location
        self.memory_ratings = []
        self.memories = []
        self.compressed_memories = []
        self.plans = ""
        self.world_graph = world_graph
        
    def __repr__(self):
        return f"Agent({self.name}, {self.description}, {self.location})"
    
    def plan(self, global_time, prompt_meta):
        """
        Generates the agent's daily plan.
        Parameters:
        -----------
        global_time : int
            The current time in the simulation.
        prompt_meta : str
            The prompt used to generate the plan.
        """

        prompt = f"You are {self.name}. The following is your description: {self.description} "
        prompt += f"You just woke up. What is your goal for today? "
        prompt += f"Write it down in an hourly basis, starting at {str(global_time)}:00. "
        prompt += f"Write only one or two very short sentences. Be very brief. Use at most 50 words."
        self.plans = generate(prompt_meta.format(prompt))
    
    def execute_action(self, other_agents, location, global_time, town_areas, prompt_meta):

        """Executes the agent's action based on their current situation and interactions with other agents.
        
        Parameters:
        -----------
        other_agents : list
            A list of other Agent objects in the simulation.
        location : Location
            The current Location object where the agent is located.
        global_time : int
            The current time in the simulation.
        town_areas : dict
            A dictionary of Location objects representing different areas in the simulated environment.
        prompt_meta : str
            The prompt used to generate the action.

        Returns:
        --------
        action : str
            The action executed by the agent.
        """

        people = [agent.name for agent in other_agents if agent.location == location]
        
        prompt = f"You are {self.name}. Your plans are: {self.plans}. "
        prompt += f"You are currently in {location.name} "
        prompt += f"with the following description: {town_areas[location.name]}. "
        prompt += f"It is currently {str(global_time)}:00. The following people are in this area: {', '.join(people)}. "
        
        people_description = [f"{agent.name}: {agent.description}" for agent in other_agents if agent.location == location.name]
        prompt += ' You know the following about people: ' + '. '.join(people_description)
        
        prompt += "What do you do in the next hour? Use at most 10 words to explain."
        action = generate(prompt_meta.format(prompt))
        return action
    
    def update_memories(self, other_agents, global_time, action_results):
        
        """
        Updates the agent's memories based on their interactions with other agents.
        
        Parameters:
        -----------
        other_agents : list
            A list of other Agent objects in the simulation.
        global_time : int
            The current time in the simulation.
        action_results : dict
            A dictionary of the results of each agent's action.
        """

        for agent in other_agents:
            if agent.location == self.location:
                self.memories.append(f'[Time: {str(global_time)}. Person: {agent.name}. Memory: {action_results[agent.name]}]\n')

    def compress_memories(self, global_time, MEMORY_LIMIT=10):

        """
        Compresses the agent's memories to a more manageable and relevant set.
        
        Parameters:
        -----------
        global_time : int
            The current time in the simulation.
        MEMORY_LIMIT : int, optional
            The maximum number of memories to compress. Default is 10.

        Returns:
        --------
        memory_string : str
            The compressed memory string.
        """

        memories_sorted = sorted(self.memory_ratings, key=lambda x: x[1], reverse=True)
        relevant_memories = memories_sorted[:MEMORY_LIMIT]
        memory_string_to_compress = '.'.join([a[0] for a in relevant_memories])
        return f'[Recollection at Time {str(global_time)}:00: {memory_string_to_compress}]'
    
    def rate_memories(self, locations, global_time, prompt_meta):

        """
         Rates the agent's memories based on their relevance and importance.
        
        Parameters:
        -----------
        locations : Locations
            The Locations object representing different areas in the simulated environment.
        global_time : int
            The current time in the simulation.
        prompt_meta : str
            The prompt used to rate the memories.

        Returns:
        --------
        memory_ratings : list
            A list of tuples representing the memory, its rating, and the generated response.
        """

        memory_ratings = []
        for memory in self.memories:
            prompt = f"You are {self.name}. Your plans are: {self.plans}. "
            prompt += f"You are currently in {locations.get_location(self.location)}. "
            prompt += f"It is currently {str(global_time)}:00. "
            prompt += f"You observe the following: {memory}. Give a rating, between 1 and 5, "
            prompt += f"to how much you care about this."
            res = get_rating_fast(prompt)
            rating = str(res)
            memory_ratings.append((memory, rating, res))
        self.memory_ratings = memory_ratings
        return memory_ratings


    def rate_locations(self, locations, global_time, prompt_meta):

        """
        Rates different locations in the simulated environment based on the agent's preferences and experiences.
        
        Parameters:
        -----------
        locations : Locations
            The Locations object representing different areas in the simulated environment.
        global_time : int
            The current time in the simulation.
        prompt_meta : str
            The prompt used to rate the locations.

        Returns:
        --------
        place_ratings : list
            A list of tuples representing the location, its rating, and the generated response.

        """

        place_ratings = []
        locations.get_location(self.location)
        location_prompt = ""
        location_symbol = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'a']
        for j, (k, v) in enumerate(locations.locations.items()):
            location_prompt+= f"{location_symbol[j]}. {k}-{v.description}"
        prompt = f"You are {self.name}. Your plans are: {self.plans}. "
        prompt += f"It is currently {str(global_time)}:00. "
        prompt += f"You are currently at {locations.get_location(self.location)}. "
        prompt += f"Which location will you go next? "
        prompt += f"Here is symbol, location, and description of locations: "
        prompt += location_prompt
        prompt += f"Answer with symbol that indicates the location."
        ratings = get_best_location_fast(prompt)
        
        for j, (k, v) in enumerate(locations.locations.items()):
            res = ratings[j]
            rating = str(res)
            place_ratings.append((k, rating, res))
        self.place_ratings = place_ratings
        return sorted(place_ratings, key=lambda x: x[1], reverse=True)
    
    def move(self, new_location_name):

        if new_location_name == self.location:
            return self.location

        try:
            path = nx.shortest_path(self.world_graph, source=self.location, target=new_location_name)
            self.location = new_location_name
        except nx.NetworkXNoPath:
            print(f"No path found between {self.location} and {new_location_name}")
            return self.location

        return self.location

