import numpy as np
import math
DT = 0.02
WB = 0.03
NX = 4
NU = 2

def unit_vector(vector):
    #Returns the unit vector of the vector. 
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    #Returns the angle in radians between vectors 'v1' and 'v2'
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# function from 
# https://github.com/AtsushiSakai/PythonRobotics/blob/eb6d1cbe6fc90c7be9210bf153b3a04f177cc138/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py#L80-L102
def get_linear_model_matrix(v, phi, delta):
    A = np.matrix(np.zeros((NX, NX)))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = - DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.matrix(np.zeros((NX, NU)))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = - DT * v * math.cos(phi) * phi
    C[3] = - v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical angle
        self.p_ang = None
        # physical velocity
        self.p_vel = None
        # steering angle
        #self.phi = None
        # physical angular velocity
        #self.p_avel = None      
        # physical acceleration
        #self.acc = None
        # physical angular acceleration
        #self.ag_acc = None   

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# state of landmark
class LandmarkState(EntityState):
    def __init__(self):
        super(LandmarkState, self).__init__()
        self.c = None


# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # physical rotated angle
        #self.phi = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # length between wheels (0.6*self.size)
        self.length = 0.05
        # rotation radius
        self.r = 3
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()
        self.movable = True
        self.silent = False
        self.blind = False
        self.u_noise = None
        self.phi_noise = None
        self.c_noise = None
        self.phi_range = 60.0
        # state
        self.state = LandmarkState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # physical rotated angle noise amount
        self.phi_noise = None
        # communication noise amount
        self.c_noise = None
        # control u range
        self.u_range = 3.0
        # control phi range
        self.phi_range = 60.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # angle dimensionality
        self.dim_ang = 1
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.05
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]
    @property
    def scripted_landmarks(self):
        return [landmark for landmark in self.landmarks if landmark.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        # set actions for scripted landmarks
        for landmark in self.scripted_landmarks:
            landmark.action = landmark.action_callback(landmark, self)
        #np.random.uniform(-1, 1) * 30 
        #entity.action.phi = phi = apply_rotation_angle(self)    
        # gather forces applied to entities
        #p_force = [None] * len(self.entities)
        # apply agent physical controls
        #p_force = self.apply_action_force(p_force)
        #print('force:',p_force)
        # apply environment forces
        # self.apply_environment_force()
        #p_force = self.apply_environment_force(p_force)
        #self.constraint_action_within_range()
        # gather agent acceleration
        #self.gather_agent_acceleration()
        # generate real action by adding noise
        #self.add_noise_in_action()
        # integrate physical state
        self.integrate_agent_state()
        self.integrate_landmark_state()
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

        # update landmark state
        for landmark in self.landmarks:
            self.update_landmark_state(landmark)
    
    # gather agent rotation angle
    # def apply_rotation_angle(self):
    #     # generate any angle betweem -10 and 10
    #     phi = np.random.uniform(-1, 1) * 30
    #     return phi

    # gather agent action forces
    # def apply_action_force(self, p_force):
    #     # set applied forces
    #     for i,agent in enumerate(self.agents):
    #         if agent.movable:
    #             noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
    #             #print("agent.action.u",agent.action.uentity.state.p_pos[0] = np.clip(entity.state.p_pos[0], -1, 1))
    #             #print("noise:",noise)entity.state.p_pos[0] = np.clip(entity.state.p_pos[0], -1, 1)
    #             p_force[i] = agent.action.u + noise   entity.state.p_pos[0] = np.clip(entity.state.p_pos[0], -1, 1)             
    #     return p_force
    
    # gather agent acceleration
    def gather_agent_acceleration(self):
        for i, agent in enumerate(self.agents):
            if agent.movable:
                agent.state.acc = agent.action.u[0]
                agent.state.ag_acc = agent.action.u[1]
                # if agent.state.acc < -3:
                #     agent.state.acc = -3#agent.state.acc
                # if agent.state.acc > 3:
                #     agent.state.acc=3
                # if agent.state.ag_acc > 2:
                #     agent.state.ag_acc=2
                # if agent.state.ag_acc < -2:
                #     agent.state.ag_acc=-2            
                #print("av:", agent.state.acc)
                #print("aw:", agent.state.ag_acc)

    def gather_landmark_acceleration(self):
        for i, landmark in enumerate(self.landmarks):
            if landmark.movable:
                landmark.state.acc = landmark.action.u[0]
                landmark.state.ag_acc = landmark.action.u[1]


    def constraint_action_within_range(self):
        for entity in self.entities:
            if not entity.movable: continue
            entity.action.u[0] = 2.5 * entity.action.u[0]    
            if entity.action.u[1] > 2.2 and entity.action.u[1] < -2.2:
                entity.action.u[1] = 2.2 * np.random.uniform(-1,1)

    # generate real action by adding noise
    def add_noise_in_action(self):
        for agent in self.agents:
            if agent.movable:
                print("velocity", agent.action.u[0])
                print("angular velocity", agent.action.u[1])
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                agent.action.u += noise

    # gather physical forces acting on entities
    def apply_environment_force(self):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(entity_a.action.u[0] is None): entity_a.action.u[0] = 0.0
                    #print("acc:",entity_a.action.u[0]) 
                    #print("f_a:",f_a)
                    #print("mass:",entity_a.mass)   
                    entity_a.action.u[0] += f_a/entity_a.mass  
                if(f_b is not None):
                    if(entity_b.action.u[0] is None): entity_b.action.u[0] = 0.0
                    entity_b.action.u[0] += f_b/entity_b.mass         

    # integrate physical state

    def integrate_agent_state(self):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            if entity.action.u is not None:

                # version: use Z_{t+1} = A * Z_{t} + B * u_{t} +C        
                # entity.action.u[0] = np.clip(entity.action.u[0], -2, 2) # acceleration
                # entity.action.u[1] = np.clip(entity.action.u[1], -math.pi/6, math.pi/6) # steering angle
                # v_total = np.sqrt(entity.state.p_vel[0]**2 + entity.state.p_vel[1]**2)   
                # phi = entity.state.p_ang
                # delta = entity.action.u[1] 
                # A, B, C = get_linear_model_matrix(v_total, phi, delta)  

            
                # current_state = np.array([entity.state.p_pos[0],
                #                           entity.state.p_pos[1],
                #                           v_total,
                #                           entity.state.p_ang]).reshape(4,1)
                # #print("current_state:",current_state)
                # current_action = np.array(entity.action.u).reshape(2,1)
                # next_state = A * current_state + B * current_action + C.reshape(4,1)
                # entity.state.p_pos[0] = next_state[0,0]
                # entity.state.p_pos[1] = next_state[1,0]
                # entity.state.p_ang = next_state[3,0]
                # v_total = next_state[2,0]
                # entity.state.p_vel[0] = v_total * np.cos(entity.state.p_ang)
                # entity.state.p_vel[1] = v_total * np.sin(entity.state.p_ang)

                # #version: update state
                # entity.action.u[0] = np.clip(entity.action.u[0], -3, 3) # acceleration
                # entity.action.u[1] = np.clip(entity.action.u[1], -math.pi/3, math.pi/3) # steering angle                
                # entity.state.p_pos[0] += entity.state.p_vel[0] * self.dt
                # entity.state.p_pos[1] += entity.state.p_vel[1] * self.dt
                # v_total = np.sqrt(entity.state.p_vel[0]**2 + entity.state.p_vel[1]**2)   
                # entity.state.p_ang += v_total / entity.length * np.tan(entity.action.u[1]) * self.dt
                # v_total += entity.action.u[0] * self.dt
                # v_total = np.clip(v_total, -2.0/3.6, 5.5/3.6) # velocity
                # entity.state.p_vel[0] = v_total * np.cos(entity.state.p_ang)
                # entity.state.p_vel[1] = v_total * np.sin(entity.state.p_ang)


                # Non holonomic dynamics
                entity.action.u[0] = np.clip(entity.action.u[0], -1., 1.) # acceleration
                entity.action.u[1] = np.clip(entity.action.u[1], -math.pi/3, math.pi/3) # steering angle 

                speed, heading = entity.state.p_vel[0], entity.state.p_vel[1]

                entity.state.p_pos[0] += speed * np.cos(heading) * self.dt
                entity.state.p_pos[1] += speed * np.sin(heading) * self.dt
                entity.state.p_vel[1] += speed / entity.length * np.tan(entity.action.u[1]) * self.dt
                #entity.state.p_vel[1] %= 2 * np.pi
                entity.state.p_vel[1] = math.asin(math.sin(entity.state.p_vel[1]))
                entity.state.p_ang = heading
                #entity.state.p_ang %= 2 * np.pi
                entity.state.p_vel[0] += entity.action.u[0] * self.dt
                #entity.state.p_vel[0] = np.clip(entity.state.p_vel[0], -2.0/3.6, 5.5/3.6)
                

                ## Holonomic
                # entity.action.u[0] = np.clip(entity.action.u[0], -3, 3) # acceleration
                # entity.action.u[1] = np.clip(entity.action.u[1], -3, 3)
                # entity.state.p_pos[0] += entity.state.p_vel[0] * self.dt
                # entity.state.p_pos[1] += entity.state.p_vel[1] * self.dt
                # entity.state.p_vel[0] += entity.action.u[0] * self.dt
                # entity.state.p_vel[1] += entity.action.u[1] * self.dt

                entity.state.p_pos[0] = np.clip(entity.state.p_pos[0], -1.5, 1.5)
                entity.state.p_pos[1] = np.clip(entity.state.p_pos[1], -1.5, 1.5)
                entity.state.p_vel[0] = np.clip(entity.state.p_vel[0], -2, 2)
                entity.state.p_vel[1] = np.clip(entity.state.p_vel[1], -2*np.pi, 2*np.pi)


                # print('!')
 
                #print("next_state",next_state)
                #print("position:",entity.state.p_pos)
                #print("velocity:",entity.state.p_vel)
                #print("acceleration:",entity.state.acc)
                #print("angle:",entity.state.p_ang)
                # entity.state.p_pos[0] += entity.state.p_vel[0] * self.dt + (1/2 * entity.state.acc * (self.dt)**2 ) * np.cos(entity.state.p_ang)
                # entity.state.p_pos[1] += entity.state.p_vel[1] * self.dt + (1/2 * entity.state.acc * (self.dt)**2 ) * np.sin(entity.state.p_ang)
                # if entity.state.ag_acc > 0:
                #     entity.state.p_avel = np.sqrt(entity.state.ag_acc/entity.r)
                #     entity.state.p_ang += entity.state.p_avel * self.dt 
                # if entity.state.ag_acc < 0:
                #     entity.state.p_avel = np.sqrt(-entity.state.ag_acc/entity.r)
                #     entity.state.p_ang -= entity.state.p_avel * self.dt 

                # if entity.state.phi > np.pi/6:
                #     entity.state.phi = np.pi/6
                # if entity.state.phi < -np.pi/6:
                #     entity.state.phi = -np.pi/6    
                # total_vel = np.sqrt(entity.state.p_vel[0]**2 + entity.state.p_vel[1]**2)    
                # entity.state.p_ang += total_vel/ entity.length * np.tan(entity.state.phi) * self.dt
                #entity.state.p_ang += entity.state.p_avel * self.dt + 1/2 * entity.state.ag_acc * (self.dt)**2

            # if entity.action.u is not None:
            #     if entity.action.u[0]>0:
            #         entity.state.p_pos[0] += entity.action.u[0] * self.dt * np.cos(entity.state.p_ang)
            #         entity.state.p_pos[1] += entity.action.u[0] * self.dt * np.sin(entity.state.p_ang)
            #         entity.state.p_ang += entity.action.u[1] * self.dt
            #     elif entity.action.u[0]<0:
            #         entity.state.p_ang = self.contraint_angle_range(entity.state.p_ang+np.pi)
            #         entity.state.p_pos[0] -= entity.action.u[0] * self.dt * np.cos(entity.state.p_ang)
            #         entity.state.p_pos[1] -= entity.action.u[0] * self.dt * np.sin(entity.state.p_ang)
            #         entity.state.p_ang += entity.action.u[1] * self.dt   
            #     else:
            #         entity.state.p_ang += entity.action.u[1] * self.dt                     
            # entity.state.p_ang = self.contraint_angle_range(entity.state.p_ang)        
            # if entity.state.p_ang <= 0:
            #     entity.state.p_ang += 2 * math.pi
            # if entity.state.p_ang >= 2 * math.pi:
            #     entity.state.p_ang -= 2 * math.pi

    def integrate_landmark_state(self):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            if entity.action.u is not None:
                # Non holonomic dynamics for landmark
                """ entity.action.u[0] = np.clip(entity.action.u[0], -1., 1.) # acceleration
                entity.action.u[1] = np.clip(entity.action.u[1], -math.pi/3, math.pi/3) # steering angle 

                speed, heading = entity.state.p_vel[0], entity.state.p_vel[1]

                entity.state.p_pos[0] += speed * np.cos(heading) * self.dt
                entity.state.p_pos[1] += speed * np.sin(heading) * self.dt
                entity.state.p_vel[1] += speed / entity.length * np.tan(entity.action.u[1]) * self.dt

                #entity.state.p_vel[1] %= 2 * np.pi
                entity.state.p_vel[1] = math.asin(math.sin(entity.state.p_vel[1]))
                entity.state.p_ang = heading
                #entity.state.p_ang %= 2 * np.pi
                entity.state.p_vel[0] += entity.action.u[0] * self.dt
                entity.state.p_pos[0] = np.clip(entity.state.p_pos[0], -1.5, 1.5)
                entity.state.p_pos[1] = np.clip(entity.state.p_pos[1], -1.5, 1.5)
                entity.state.p_vel[0] = np.clip(entity.state.p_vel[0], -2, 2)
                entity.state.p_vel[1] = np.clip(entity.state.p_vel[1], -2*np.pi, 2*np.pi) """


                entity.state.p_pos[0] += 0.1
                entity.state.p_pos[1] += 0.1
                entity.state.p_vel[1] += 0.1
                entity.state.p_ang = 0.1
                entity.state.p_vel[0] += 0.1

    def contraint_angle_range(self, theta):
        if theta <= 0:
            theta += 2*np.pi
        if theta >= 2*np.pi:
            theta -= 2*np.pi 
        return theta    
    # integrate physical state
    # def integrate_state(self, p_force):
    #     for i,entity in enumerate(self.entities):
    #         if not entity.movable: continue
    #         #entity.action.vel = entity.action.vel * (1 - self.damping)
    #         entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
    #         if (p_force[i] is not None):
    #             #entity.action.vel += (p_force[i] / entity.mass) * self.dt
    #             p_force_total = np.sqrt(p_force[i][0]**2 + p_force[i][1]**2)
    #             p_force[i][0], p_force[i][1] = p_force_total*np.cos(entity.state.p_ang/180*np.pi), p_force_total*np.sin(entity.state.p_ang/180*np.pi)
    #             entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
    #         #speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
    #         #entity.state.p_vel[0], entity.state.p_vel[1] = speed*np.cos(entity.state.p_ang/180*np.pi), speed*np.sin(entity.state.p_ang/180*np.pi)       
    #         if entity.max_speed is not None:
    #             #speed = np.sqrt(np.square(entity.action.vel[0]) + np.square(entity.action.vel[1]))
    #             speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
    #             if speed > entity.max_speed:
    #                 entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
    #                                                               np.square(entity.state.p_vel[1])) * entity.max_speed
    #         #entity.state.p_pos += entity.action.vel * self.dt
    #         entity.state.p_pos += entity.state.p_vel * self.dt
    #         entity.state.p_ang += np.sqrt(entity.state.p_vel[0]**2 + entity.state.p_vel[1]**2) * np.tan(entity.action.phi/180*np.pi) * self.dt / entity.length
    #         if entity.state.p_ang <= 0:
    #             entity.state.p_ang += 2 * math.pi
    #         if entity.state.p_ang >= 2 * math.pi:
    #             entity.state.p_ang -= 2 * math.pi

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    def update_landmark_state(self, landmark):
        if not landmark.silent:
            landmark.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*landmark.action.c.shape) * landmark.c_noise if landmark.c_noise else 0.0
            landmark.state.c = landmark.action.c + noise   

    #get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force = np.sqrt(np.sum(np.square(force)))
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        #print("force_a",force_a)
        #print("force_b",force_b)        
        return [force_a, force_b]