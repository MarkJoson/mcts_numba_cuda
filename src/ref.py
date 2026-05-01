# Author: Kyle Kastner
# License: BSD 3-Clause

# based on minigo implementation
# https://github.com/tensorflow/minigo/blob/master/mcts.py 
# Useful discussion of the benefits
# http://www.moderndescartes.com/essays/agz/

# See survey
# http://mcts.ai/pubs/mcts-survey-master.pdf
# See similar implementation here
# https://github.com/junxiaosong/AlphaZero_Gomoku

# some changes from high level pseudo-code in survey

import numpy as np
import copy
import collections

empty_board = np.zeros((3, 3))

class TicTacToeManager(object):
    def __init__(self, state=None, rollout_limit=1000):
        self.random_state = np.random.RandomState(1999)
        self.rollout_limit = rollout_limit
        if state == None:
            state = self.get_init_state()
        self.state = state

    def get_next_state(self, state, action):
        p1_c = len(np.where(state == -1.)[0])
        p2_c = len(np.where(state == 1.)[0])
        shp = state.shape
        s = state.flatten()
        if np.sum(s) == 0:
           s[action] = -1.
        else:
           s[action] = 1.
        return s.reshape(*shp)

    def get_current_player(self, state):
        if np.sum(state) == 0.:
            # equal number of +1 and -1 means its player 0 turn
            return 0
        else:
            return 1

    def get_action_space(self):
        return list(range(len(empty_board.ravel())))

    def get_valid_actions(self, state):
        return list(np.where(state.ravel() == 0)[0])

    def get_init_state(self):
        return copy.deepcopy(empty_board)

    def _rollout_fn(self, state):
        return self.random_state.choice(self.get_valid_actions(state))

    def rollout_from_state(self, state):
        # score is state relative
        # e.g. win or loss from random rollout on this state
        s = state
        w, e = self.is_finished(state)
        if e:
            if w == -1:
                return 0.
            else:
                return 1.

        c = 0
        while True:
            a = self._rollout_fn(s)
            s = self.get_next_state(s, a)
            w, e = self.is_finished(s)
            c += 1
            if e:
                if w == -1:
                    return 0

                if c % 2 == 0:
                    return 1.
                else:
                    return -1.

            if c > self.rollout_limit:
                return 0.

    def is_finished(self, state):
        # returns 0 for player 0 win
        # returns 1 for player 1 win
        # returns -1 for tie or not finished
        for i in range(3):
            if np.sum(state[i]) == 3.:
                return 1, True
            elif np.sum(state[i]) == -3.:
                return 0, True
            elif np.sum(state[:, i]) == 3.:
                return 1, True
            elif np.sum(state[:, i]) == -3.:
                return 0, True
        if np.sum(state[[0, 1, 2], [0, 1, 2]]) == 3.:
            return 1, True
        elif np.sum(state[[0, 1, 2], [0, 1, 2]]) == -3.:
            return 0, True
        elif np.sum(state[[2, 1, 0], [0, 1, 2]]) == 3.:
            return 1, True
        elif np.sum(state[[2, 1, 0], [0, 1, 2]]) == -3.:
            return 0, True
        elif len(np.where(state == 0.)[0]) == 0:
            return -1, True
        return -1, False


class EmptyNode(object):
    """Empty node of MCTS tree, placeholder for root.

    Code becomes simpler if all nodes have parents"""
    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)


class MCTSNode(object):
    """Node of MCTS tree, can compute action scores of all children
    state_manager: a state manager instance which correspnds to this node
    parent: A parent MCTSNode (None means this is a "first action" node
    action_to_here: action that led to this node, usually an integer but depends on state_manager implementation (see get_action_space / get_valid_actions)
    """
    def __init__(self, state_manager, parent=None, single_player=False, q_init=None, current_state=None, action_to_here=None, n_playout=1000, random_state=None):
        # this assumes full observability?
        if parent is None:
            parent = EmptyNode()
            action_to_here = None
            current_state = state_manager.get_init_state()
        self.single_player = single_player
        self.q_init = q_init
        if single_player and q_init is None:
            raise ValueError("Single player nodes require q_init argument to be passed")
        self.state_manager = state_manager
        self.current_state = current_state

        self.valid_actions = self.state_manager.get_valid_actions(self.current_state)
        self.illegal_moves = np.array([0. if n in self.valid_actions else 1. for n in range(len(self.state_manager.get_action_space()))])

        self.parent = parent
        self.action_to_here = action_to_here
        self.c_puct = 1.4
        self.n_playout = n_playout
        self.warn_at_ = 10000
        if random_state is None:
            raise ValueError("Must pass random_state object")
        self.random_state = random_state

        self.is_expanded = False
        self.losses_applied = 0
        # child_() allows vectorized computation of action score
        self.child_N = np.zeros([len(self.state_manager.get_action_space())], dtype=np.float32)
        self.child_W = np.zeros([len(self.state_manager.get_action_space())], dtype=np.float32)

        # do we need child priors as in the minigo code

        if q_init is None:
            self.q_init = np.zeros([len(self.state_manager.get_action_space())], dtype=np.float32)
        elif hasattr(q_init, "__len__"):
            self.q_init = np.array(q_init).astype("float32")
        else:
            # single float in
            self.q_init = np.zeros([len(self.state_manager.get_action_space())], dtype=np.float32)
            self.q_init += q_init
        self.children = {} # move map to resulting node

    def __repr__(self):
        if self.single_player:
            return "<MCTSNode move={}, N={}>".format(self.action_to_here, self.N)
        else:
            this_player = self.state_manager.get_current_player(self.current_state)
            return "<MCTSNode move={}, N={}, to_play={}>".format(self.action_to_here, self.N, this_player)

    def is_leaf(self):
        return self.children == {}

    @property
    def child_Q(self):
        child_N_nonzeros = np.where(self.child_N != 0.)[0]
        # start out_buffer with q_init values
        out_buffer = self.q_init
        out_buffer[child_N_nonzeros] = self.child_W[child_N_nonzeros] / self.child_N[child_N_nonzeros]
        return out_buffer

    @property
    def child_U(self):
        return self.c_puct * np.sqrt(self.N) / (self.child_N + 1)

    @property
    def child_action_score(self):
        return (self.child_Q + self.child_U)

    @property
    def N(self):
        return self.parent.child_N[self.action_to_here]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.action_to_here] = value

    @property
    def W(self):
        return self.parent.child_W[self.action_to_here]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.action_to_here] = value

    def get_best(self, random_tiebreak=True):
        scores = self.child_action_score
        valid_actions = [pos for pos in np.argsort(scores)[::-1] if pos in self.children.keys()]
        valid_scores = scores[valid_actions]
        if random_tiebreak:
        # random tiebreaker
            max_score = max(valid_scores)
            assert len(valid_actions) == len(valid_scores)
            equivalent_valid_scores = [(vs, va) for (vs, va) in zip(valid_scores, valid_actions) if vs == max_score]
            pair = random_state.choice(np.arange(len(equivalent_valid_scores)))
            v_a = equivalent_valid_scores[pair][1]
            child = self.children[v_a]
        else:
           v_a = valid_actions[0]
           child = self.children[v_a]
        return v_a, child

    def add_virtual_loss(self, up_to):
        """propagate virtual loss upward (to root)

        up_to, node to propagate until (track this)
        """
        self.losses_applied += 1
        if self.single_player:
            # 1 player
            # player == state related?
            loss = 1
            self.W += loss
            if self.parent is None or self is up_to:
                return
            self.parent.add_virtual_loss(up_to)
        else:
            # 2 player
            this_player = self.state_manager.get_current_player(self.current_state)
            # use this to get alternating in 2 player
            loss = -1 if this_player == 0 else 1
            self.W += loss
            if self.parent is None or self is up_to:
                return
            self.parent.add_virtual_loss(up_to)

    def revert_virtual_loss(self, up_to):
        """undo virtual losses

        up_to, node that was propagated until
        """
        self.losses_applied -= 1
        if self.single_player:
            # 1 player
            loss = 1
            revert = -1 * loss
            self.W += revert
            if self.parent is None or self is up_to:
                return
            self.parent.revert_virtual_loss(up_to)
        else:
            # 2 player
            this_player = self.state_manager.get_current_player(self.current_state)
            # use this to get alternating in 2 player
            loss = -1 if this_player == 0 else 1
            revert = -1 * loss
            self.W += revert
            if self.parent is None or self is up_to:
                return
            self.parent.revert_virtual_loss(up_to)

    def multileaf_safe_backup_value(self, value, up_to):
        # if for some reason we selected a leaf multiple times despite
        # virtual loss, don't re-run it
        if self.is_expanded:
            return

        self.is_expanded = True
        self.backup_value(value, up_to=up_to)

    def backup_value(self, value, up_to):
        if self.parent != None:
            if not (self is up_to):
                if self.single_player:
                    self.parent.backup_value(value, up_to)
                else:
                    self.parent.backup_value(-value, up_to)
        self.N += 1
        self.W += value

    def maybe_add_children(self, actions_and_probs):
        for elem in actions_and_probs:
            self.maybe_add_child(elem)

    def maybe_add_child(self, action_and_prob):
        action = action_and_prob[0]
        prob = action_and_prob[1]
        if action not in self.children:
            # need the state itself
            state = self.current_state
            next_state = self.state_manager.get_next_state(state, action)
            seed = self.random_state.randint(0, 1E6)
            rs = np.random.RandomState(seed)
            if self.single_player:
                # this will come from a neural network at some point
                q_init = self.child_Q
            else:
                # assuming 2 player, 0 sum 
                q_init = 0.
            self.children[action] = MCTSNode(self.state_manager, single_player=self.single_player, q_init=q_init, current_state=next_state, action_to_here=action, parent=self, random_state=rs)
        return self.children[action]


class MCTSPlayer(object):
    def __init__(self, state_manager, single_player=False, n_readouts=100, random_state=None):
        self.state_manager = state_manager
        self.root = MCTSNode(state_manager, parent=None, random_state=random_state)
        self.single_player = single_player
        if self.single_player:
            # will come from NN eventually, for now set simply
            # first state has default 0 value...
            q_init = 0. * np.array(self.state_manager.get_valid_actions(self.state_manager.get_init_state()))
        else:
            q_init = 0.
        self.root = MCTSNode(state_manager, single_player=self.single_player, q_init=q_init, parent=None, random_state=random_state)
        self.n_readouts = n_readouts

    def select_leaf(self):
        node = self.root
        state = self.root.current_state
        while True:
            # if node has never been expanded, don't select a child
            if node.is_leaf():
                break

            # this will need to do model evaluation
            # can we cache the evaluations?
            action, node = node.get_best()
            state = node.state_manager.get_next_state(state, action)
        return node

    def tree_search(self):
        # single leaf, non-parallel tree search
        # useful sanity check / test case
        node = self.select_leaf()
        state = node.current_state
        winner, end = node.state_manager.is_finished(state)
        if not end:
            actions = node.state_manager.get_valid_actions(state)
            action_space = node.state_manager.get_action_space()

            # uniform prior probs, zero out invalid actions
            probs = np.zeros((len(action_space),))
            probs[actions] = np.ones((len(actions))) / float(len(actions))
            # only include valid actions to the child nodes!
            actions_and_probs = list(zip(actions, probs[actions]))
            node.maybe_add_children(actions_and_probs)
        # random rollout
        value = node.state_manager.rollout_from_state(state)
        node.backup_value(value, up_to=self.root)
        return None

    def parallel_tree_search(self, parallel_readouts):
        assert parallel_readouts > 0
        leaves = []
        failsafe = 0

        while len(leaves) < parallel_readouts and failsafe < (2 * parallel_readouts):
            failsafe += 1
            # leaf selection will be NN based eventually
            leaf = self.select_leaf()

            winner, end = leaf.state_manager.is_finished(leaf.current_state)
            if end:
                # value will be predicted from NN
                value = leaf.state_manager.rollout_from_state(leaf.current_state)
                leaf.backup_value(value, up_to=self.root)
                # go back to getting leaves
                continue
            leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)

        if leaves:
            # get move probs and values
            # in original code, network predicted 
            #     move_probs, values = self.network.run_many(
            #         [leaf.position for leaf in leaves])
            action_space = self.root.state_manager.get_action_space()
            move_probs = [np.zeros((len(action_space),)) for l in leaves]
            values = []
            for n, leaf in enumerate(leaves):
                # handle per-position invalid moves
                actions = leaf.state_manager.get_valid_actions(leaf.current_state)
                # uniform probs (probs not currently used)
                move_probs[n][actions] = np.ones((len(actions))) / float(len(actions))
                # value will be predicted from NN
                value = leaf.state_manager.rollout_from_state(leaf.current_state)
                # only include valid actions in the child nodes
                actions_and_probs = list(zip(actions, move_probs[n][actions]))
                leaf.maybe_add_children(actions_and_probs)
                values.append(value)

            for leaf, move_prob, value in zip(leaves, move_probs, values):
                leaf.revert_virtual_loss(up_to=self.root)
                leaf.multileaf_safe_backup_value(value, up_to=self.root)
        return None

    def simulate(self, parallel=0):
        # not true parallelism yet, but batch evaluation should allow
        # parallelism to be possible
        assert parallel >= 0
        current_readouts = self.root.N
        while self.root.N < current_readouts + self.n_readouts:
            if parallel == 0:
                self.tree_search()
            else:
                self.parallel_tree_search(parallel_readouts=parallel)
        return None

    def pick_move(self):
        # pick a move from the set of the best
        # currently based purely on visit count
        illegal_moves = np.where(self.root.illegal_moves == 1.)[0]
        argsort_best = np.argsort(self.root.child_N)[::-1]
        valid_actions = [a for a in argsort_best if a not in illegal_moves]
        return valid_actions[0]

    def play_move(self, move):
        self.root = self.root.maybe_add_child((move, 1.))
        return True



human_opponent = False
parallel = 32
player_turn = 0
turn = 0
count = 0
# 100 readouts results in mostly player 0 wins, some ties
# 1,000 was enough with q_init=inf, but with q_init=0 ~20% lost games
# need a LOT more readouts if not using q_init=inf in each MCTSNode
# 5,000 is enough to tie all 10 games
# could preserve tree to reduce chance of game loss / combine experience
# (not feasible in large state spaces due to memory requirements)
readouts = 5000
n_games = 10

outcomes = []

for i in range(n_games):
    start = True
    if i == 0:
        # deterministic first game for debug
        random_state = np.random.RandomState(1234)
    else:
        random_state = np.random.RandomState(random_state.randint(100000))
    # set up the MCTS player and game
    sm = TicTacToeManager()
    mcts = MCTSPlayer(state_manager=sm, n_readouts=readouts, random_state=random_state)

    # play the game out
    while True:
        print(mcts.root.current_state)
        print("")
        print("Player {}".format(turn))
        print("")

        # need to do noise injection to add randomness
        # for collecting diverse games
        if not human_opponent:
            # for now, we don't have a noise injection function
            # so skip this
            pass
            #mcts.root.inject_noise()

        mcts.simulate(parallel=parallel)
        move = mcts.pick_move()
        if start:
           moves = [0, 2, 4, 6, 8]
           random_state.shuffle(moves)
           move = moves[0]
           start = False

        # for further game diversity, sample moves 
        # at some point down the game tree
        # as in AlphaZero
        if turn != player_turn:
            mcts.play_move(move)
            turn = int(not (bool(turn) ^ bool(0)))
        else:
            if human_opponent and turn == player_turn:
                print("Input move action, 0-8:")
                inp = raw_input()
                if inp[0] == "d":
                    print("debug")
                    from IPython import embed; embed(); raise ValueError()
                move = int(inp)
                mcts.play_move(move)
                turn = int(not (bool(turn) ^ bool(0)))
            else:
                mcts.play_move(move)
                turn = int(not (bool(turn) ^ bool(0)))
        score, finished = mcts.root.state_manager.is_finished(mcts.root.current_state)
        count += 1
        if finished:
            print(mcts.root.current_state)
            print("Game {} over".format(i))
            if score == 0:
                print("Winner, player 0")
                outcomes.append("W")
            elif score > 0:
                print("Winner, player 1")
                outcomes.append("L")
            else:
                print("Tie game")
                outcomes.append("T")
            print("")
            break

c = len(outcomes)
w_count = sum([o == "W" for o in outcomes])
l_count = sum([o == "L" for o in outcomes])
t_count = sum([o == "T" for o in outcomes])

print("Stats from player 0 perspective")
print("Wins (percentage): {} ({:.2f})".format(w_count, 100 * w_count / float(c)))
print("Losses (percentage): {} ({:.2f})".format(l_count, 100 * l_count / float(c)))
print("Ties (percentage): {} ({:.2f})".format(t_count, 100 * t_count / float(c)))
print("Total games: {}".format(c))
print("Games complete! Dropping to embed")
from IPython import embed; embed(); raise ValueError()