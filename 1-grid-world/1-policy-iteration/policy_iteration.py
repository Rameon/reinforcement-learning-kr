# -*- coding: utf-8 -*-
import random

# environment.py 안에 있는 GraphicDisplay, Env 클래스를 import함
# GraphicDisplay 는 GUI로 그리드월드 환경을 보여주는 클래스임
from environment import GraphicDisplay, Env


# 정책 이터레이션의 에이전트는 PolicyIteration 클래스로 정의되어 있음
class PolicyIteration:
    def __init__(self, env):  # PolicyIteration 클래스의 정의에서 env를 self.env로서 정의함
        # 환경에 대한 객체 선언
        # 에이전트에게는 환경에 대한 정보가 필요하므로 main 루프에서 Env()를 env 객체로 생성함
        # 이 env 객체를 PolicyIteration 클래스의 인수로 전달함으로써, 에이전트는 환경의 Env() 클래스에 접근할 수 있음
        self.env = env

        # < env 객체에 정의돼 있는 변수와 함수 >
        # env.width, env.height : 그리드 월드의 너비, 높이 -> 반환값 : 그리드월드의 가로, 세로를 정수로 반환함

        # env.state_after_action(state, action) : 에이전트가 특정 상태에서 특정 행동을 했을 때, 에이전트가 가는 다음 상태
        #                                         -> 반환값 : 행동 후의 상태를 좌표로 표현한 리스트를 반환함 예) [1,2]

        # DP에서는 에이전트가 모든 상태에 대해 벨만 방정식을 계산하는데, 따라서 에이전트는 가능한 모든 상태를 알아야 함
        # 이 모든 상태들은 env.get_all_state() 를 통해 할 수 있음

        # env.get_all_state() : 존재하는 모든 상태 -> 반환값 : 모든 상태를 반환함 예) [[0,0], [0,1], ... , [4,4]]

        # env.get_reward(state, action) : 특정 상태의 보상(환경이 주는 보상) -> 반환값 : 정수의 형태로 보상을 반환함

        # env.possible_actions : 상, 하, 좌, 우(에이전트의 가능한 모든 행동) -> 반환값 : [0,1,2,3]을 반환함, 순서대로 상,하,좌,우를 의미함

        # 보상과 상태 변환 확률은 에이전트가 아니라, 환경에 속한 것이므로 env 객체로 정의함

        # 가치함수를 2차원 리스트로 초기화
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        # 상 하 좌 우 동일한 확률로 정책 초기화
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width
                                    for _ in range(env.height)]
        # 마침 상태의 설정
        self.policy_table[2][2] = []
        # 감가율
        self.discount_factor = 0.9

    # 정책 평가
    # DP에서는 사용자가 주는 입력에 따라 에이전트가 역할을 수행하기 때문에,
    # 에이전트는 environment.py의 GraphicDisplay 클래스에서 실행됨
    # 따라서 GraphicDisplay 클래스는 PolicyIteration 클래스의 객체인 policy_iteration을 상속받음
    def policy_evaluation(self):

        # 다음 가치함수 초기화
        next_value_table = [[0.00] * self.env.width
                                    for _ in range(self.env.height)]

        # 모든 상태에 대해서 벨만 기대방정식을 계산
        for state in self.env.get_all_states():
            value = 0.0
            # 마침 상태의 가치 함수 = 0
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = value
                continue

            # 벨만 기대 방정식
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value += (self.get_policy(state)[action] *
                          (reward + self.discount_factor * next_value))

            next_value_table[state[0]][state[1]] = round(value, 2)

        self.value_table = next_value_table

    # 현재 가치 함수에 대해서 탐욕 정책 발전
    def policy_improvement(self):
        next_policy = self.policy_table
        for state in self.env.get_all_states():
            if state == [2, 2]:
                continue
            value = -99999
            max_index = []
            # 반환할 정책 초기화
            result = [0.0, 0.0, 0.0, 0.0]

            # 모든 행동에 대해서 [보상 + (감가율 * 다음 상태 가치함수)] 계산
            for index, action in enumerate(self.env.possible_actions):
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                temp = reward + self.discount_factor * next_value

                # 받을 보상이 최대인 행동의 index(최대가 복수라면 모두)를 추출
                if temp == value:
                    max_index.append(index)
                elif temp > value:
                    value = temp
                    max_index.clear()
                    max_index.append(index)

            # 행동의 확률 계산
            prob = 1 / len(max_index)

            for index in max_index:
                result[index] = prob

            next_policy[state[0]][state[1]] = result

        self.policy_table = next_policy

    # 특정 상태에서 정책에 따른 행동을 반환
    def get_action(self, state):
        # 0 ~ 1 사이의 값을 무작위로 추출
        random_pick = random.randrange(100) / 100

        policy = self.get_policy(state)
        policy_sum = 0.0
        # 정책에 담긴 행동 중에 무작위로 한 행동을 추출
        for index, value in enumerate(policy):
            policy_sum += value
            if random_pick < policy_sum:
                return index

    # 상태에 따른 정책 반환
    def get_policy(self, state):
        if state == [2, 2]:
            return 0.0
        return self.policy_table[state[0]][state[1]]

    # 가치 함수의 값을 반환
    def get_value(self, state):
        # 소숫점 둘째 자리까지만 계산
        return round(self.value_table[state[0]][state[1]], 2)


if __name__ == "__main__":
    env = Env()
    policy_iteration = PolicyIteration(env)
    grid_world = GraphicDisplay(policy_iteration)
    grid_world.mainloop()
