# Introduction to Modiste
Modiste is an interactive tool designed to enhance decision-making by personalizing decision support for each user. It recognizes that individuals benefit from different forms of support based on their expertise and context. For instance, a radiologist might perform better with model predictions, while another might prefer insights from peers. Modiste dynamically tailors these recommendations to suit individual needs.

Leveraging stochastic contextual bandit algorithms, Modiste learns personalized decision support policies in real time, even without prior information about the user. It accounts for multi-objective settings, balancing decision accuracy with support costs. Validated through computational and human subject experiments on vision and language tasks, Modiste demonstrates significant improvements over static approaches.

Below are the key steps in Modiste’s workflow:
Modiste frames the decision-making process as a stochastic contextual bandit problem, where:

Forms of Support (Arms): Different types of assistance available to decision-makers (e.g., model predictions, expert consensus).
Context Space (X): The information associated with a specific decision.
The objective is to learn a decision support policy that minimizes prediction error while ensuring cost-efficiency in a cost-aware setting.

# Learning Decision Support Policies
To develop personalized policies, Modiste uses two primary approaches:

LinUCB:

Approximates prediction errors using a linear function 

Updates parameters using human feedback and normalizes error estimates to a range of [0, 1].
K-Nearest Neighbors (KNN):

Maintains a data buffer of historical interactions and estimates prediction errors by averaging the errors of 
𝐾
K-nearest neighbors.
Both methods incorporate an exploration bonus to guide policy updates and ensure optimal learning within limited interactions.

# Cost-Aware Decision Support
To account for the expense of providing support, Modiste integrates cost considerations into its optimization process. The objective becomes: λ controls the trade-off between minimizing loss and cost.

This enables Modiste to select policies that achieve high performance while minimizing resource expenditure, even in cases with strict cost constraints.

# Interactive User Interface
Modiste’s interface provides an extendable platform to study and deploy decision support policies. It offers three primary forms of support:

Human Alone: The user makes decisions without external aid.
Model Prediction: Displays AI-generated predictions.
Expert Consensus: Shows a distribution of labels derived from multiple annotators.
At each time step, Modiste updates the interface based on user feedback and selected support, enabling users to learn the effectiveness of different forms of assistance. The tool integrates seamlessly with platforms like Prolific, making it adaptable for real-world applications.

Modiste empowers responsible AI deployment by tailoring decision support to individual users. It ensures that the right form of assistance is provided at the right time, balancing accuracy and cost. Through its innovative learning algorithms and interactive interface, Modiste sets a new standard for adaptive and personalized decision-making tools.

# Steps to run modiste:
1. Run generate_env_requirements.py and create an env with dependencies.
2. Run server.py in the created environment.
3. check server's response using something like this - curl "http://localhost:80/user1244*1*C*B*0*algLinUCB_0.9"
4. if above steps work, backend is working.
5. for front end, start with creating a pavlovia server and leave it running.
6. Then go to real_index.html and check and put pavlovia link accordingly if required.
7. similarly go to config.json and put your pavlovia credentials as well.
8. run task.js and your front end should be up and running as well.
9. Reach out in case of any errors. 


