# *`Agent Designs`*
This repository contains an implementation of the prominent agentic designs like reflection, tool, planning, multiagent using lightweight custom agents and function decorators.

## What This Project Includes

- **Reflection Agent** for code generation and improvement
- **Planning Agent (ReAct)** for step-wise mathematical reasoning
- **Tool Agent (Restaurant Finder)** using OpenStreetMap & Overpass API
- **Multi-Agent Workflow**:
  - Poem generation
  - Translation 
  - Save outputs to `.txt` files

Note: This project works with Groq as the LLM provider, so you'll need to create an API Key on Groq.

---
## Installation
```sh
git clone https://github.com/shifs999/Agent_Designs.git
cd agent_designs
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
---

## Agents Overview

### 1. `Reflection Agent`: Code Generation + Refinement

* **Purpose**: Generates clean, efficient code and refines it over multiple reasoning steps.
* **Inputs**:

  * `user_msg`: coding task (e.g., “Implement any xyz algorithm”)
  * `generation_system_prompt`: coding role (e.g., Python developer)
  * `reflection_system_prompt`: reviewer role (e.g., algorithm and engineer)
* **Output**: Final improved code after iterative self-review
* **Usage**:

```python
Eg. (user_msg="Implement DFS", generation_system_prompt=..., reflection_system_prompt=..., n_steps=10)
```

### 2. `Planning Agent`: Basic Mathematical problem solver

* **Purpose**: Solves compound math queries using tool functions (`sum`, `multiply`, `log`) with stepwise reasoning.
* **Workflow**: Thought -> Tool Call -> Observation -> Final Answer
* **Tools**:

  * `sum_two_elements(a, b)`
  * `multiply_two_elements(a, b)`
  * `compute_log(x)`
* **Usage**:

```python
Eg. (user_msg="Add 123 and 456, multiply the result by 5, then take log")
```

### 3. `Tool Agent`: Restaurant Finder via OpenStreetMap

* **Purpose**: Fetches top nearby restaurants using location, metadata richness, and optional food type filters (veg/non-veg).
* **Tool**: `fetch_osm_restaurants(hotspot_name: str, top_n: int)`
* **Output**: List of restaurants with address, contact, hours, location
* **Usage**:

```python
(user_msg="Top 5 restaurants near Shibuya, Tokyo")
```

### 4. `Multi-Agent`: Translation, Poem Generation, File Saving

* **Purpose**: A pipeline of agents for multilingual tasks and creative writing
* **Workflow**:

  1. Generate poem
  2. Translate to target language
  3. Save to `.txt` file
* **Tools**:

  * `generate_poem(topic: str)`
  * `translate_text(text: str, lang: str)`
  * `write_str_to_txt(text: str, filename: str)`
* **Usage**:

```python
(user_msg="Write a poem on nature, translate to French, and save as .txt file in xyz location")
```

---

## Contributions 

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## Contact

For any queries or collaborations, feel free to reach me out at **saizen777999@gmail.com**
