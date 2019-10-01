import collections
import collections.abc
import math

EPSILON="ε" # Special symbol that is treated as the empty string
STOP="</s>" # Special symbol that occurs at the end of every string

def logaddexp(lx, ly):
    """Compute log(exp(lx)+exp(ly)) in a numerically stable way."""
    if lx < ly: lx, ly = ly, lx
    return lx + math.log1p(math.exp(ly-lx))

class Transition(collections.namedtuple('Transition', 'q a b r')):
    """A transition of a FST.
    
              a:b
           q -----> r

    `q`: The state that the transition goes from. It can be any kind of
       object, as long as it's hashable.
    `a`: The input symbol.
    `b`: The output symbol.
    `r`: The state that the transition goes to.

    Transitions are hashable and can be compared using `==` and `!=`.

    You can add whatever other attributes you want, but they will not
    be used when hashing or comparing.
    """

def _lambda_defaultdict_set():
    """Lift lambda: defaultdict(set) to global scope to enable pickling."""
    return collections.defaultdict(set)
    
class FST(object):
    """A finite state transducer."""
    
    def __init__(self):
        defaultdict = collections.defaultdict
        self.states = set()
        self.transitions = defaultdict(_lambda_defaultdict_set)
        self.transitions_inverse = defaultdict(_lambda_defaultdict_set)
        self.start = None
        self.accept = None
        self.input_alphabet = set()
        self.output_alphabet = set()
        self.counts = collections.Counter()

    def add_state(self, q):
        """Adds state q."""
        self.states.add(q)

    def set_start(self, q):
        """Sets the start state to q. If it's not already a state, it is added
        automatically."""
        self.add_state(q)
        self.start = q

    def get_start(self):
        """Returns the start state."""
        return self.start

    def set_accept(self, q):
        """Sets the accept state to q. If it's not already a state, it is added
        automatically."""
        self.add_state(q)
        self.accept = q

    def get_accept(self):
        """Returns the accept state."""
        return self.accept

    def add_transition(self, t, count=1):
        """Adds transition `t` to FST `self`.
        
        If `t.q` and `t.r` are not already states, they are added
        automatically.

        Increments the count of `t` by `count` (default 1).
        """

        if t not in self.counts:
            self.add_state(t.q)
            self.add_state(t.r)
            self.input_alphabet.add(t.a)
            self.output_alphabet.add(t.b)
            self.transitions[t.q][t.a].add(t)
            self.transitions_inverse[t.q][t.b].add(t)
        self.counts[t] += count

    def get_transitions(self, q, a):
        """Returns the set of transitions from state `q` on input `a`."""
        return self.transitions[q][a]

    def _ipython_display_(self):
        """Draw transition diagram as SVG for display in a Jupyter notebook."""
        import IPython.display
        import subprocess
        import html

        dot = ['digraph {',
               '  rankdir=LR;',
               '  node [fontname=Courier,fontsize=10];',
               '  node [shape=box,style=rounded];',
               '  node [height=0,width=0,margin="0.055,0.042"];',
               '  edge [arrowhead=vee,arrowsize=0.5];',
               '  edge [fontname=Courier,fontsize=9];',
               '  START [label="",shape=none];']
        index = {}
        for i, q in enumerate(self.states):
            index[q] = i
            attrs = 'label=<{}>'.format(html.escape(str(q)))
            if q == self.accept:
                attrs += ',peripheries=2'
            dot.append('  {} [{}];'.format(i, attrs))
        dot.append('  START->{};'.format(index[self.start]))
        for q in self.states:
            ts = collections.defaultdict(list)
            for a in self.transitions.get(q, {}):
                for t in self.transitions[q].get(a, []):
                    ts[t.r].append(t)
            for r in ts:
                label = ['<table border="0" cellpadding="1">']
                for i, t in enumerate(ts[r]):
                    if i >= 3:
                        label.append('<tr><td>...</td></tr>')
                        break
                    l = '{}:{}'.format(t.a, t.b)
                    label.append('<tr><td>{}</td></tr>'.format(html.escape(l)))
                label.append('</table>')
                dot.append('  {}->{}[label=<{}>];'.format(index[q],
                                                          index[r],
                                                          ''.join(label)))
        dot.append('}')
        dot = '\n'.join(dot)
        proc = subprocess.run(['dot', '-T', 'svg'],
                              input=dot.encode('utf8'),
                              capture_output=True,
                              check=True)
        IPython.display.display(IPython.display.SVG(proc.stdout))

def string(w):
    """Returns a FST that accepts the language {w} and maps w to itself."""
    m = FST()
    m.set_start(0)
    for i, a in enumerate(w):
        t = Transition(i, a, a, i+1)
        m.add_transition(t)
    t = Transition(len(w), STOP, STOP, len(w)+1)
    m.add_transition(t)
    m.set_accept(len(w)+1)
    return m

def estimate_joint(counts, add=0):
    """Computes joint probabilities P(a, b | q)."""
    probs = {}
    z = collections.Counter()
    for t in counts:
        z[t.q] += counts[t]+add
    for t in counts:
        probs[t] = (counts[t]+add)/z[t.q]
    return probs

def estimate_cond(counts, add=0):
    """Computes conditional probabilities P(b | q, a).

    Unlike the definition in the class notes, this does allow a
    state to have outgoing epsilon and non-epsilon transitions.
    """
    probs = {}
    s = collections.Counter()
    z = collections.Counter()
    for t in counts:
        wt = counts[t]+add
        s[t.q,t.a] += wt
        z[t.q] += wt
    for t in counts:
        wt = counts[t]+add
        if t.a == EPSILON:
            probs[t] = wt/z[t.q]
        else:
            probs[t] = wt/s[t.q,t.a]*(1-s[t.q,EPSILON]/z[t.q])
    return probs

class dynamic_values(collections.abc.Mapping):
    """A dict-like object such that `dynamic_values(f)[t] == f(t)`."""
    def __init__(self, f):
        self.f = f
    def __getitem__(self, t):
        return self.f(t)
    def __iter__(self): raise NotImplementedError()
    def __len__(self): raise NotImplementedError()
    
class ComposedFST(FST):
    def compose_values(self, v1, v2, op='*'):
        """Return a dict-like object that computes values on-the-fly from
        values (`v1` and `v2`) of the two component FSTs.
        
        `op`: either `"*"` or `"+"` indicating whether values should be
        multiplied or added."""
        return compose_values(self, v1, v2, op)

class compose_values(collections.abc.Mapping):
    def __init__(self, m, v1, v2, op='*'):
        self.m = m
        self.v1 = v1
        self.v2 = v2
        self.op = op

    def __getitem__(self, t):
        if self.op == '*':
            v = 0.
            for (t1, t2) in self.m.composed_from[t]:
                dv = 1.
                if t1 is not None: dv *= self.v1[t1]
                if t2 is not None: dv *= self.v2[t2]
                v += dv
        elif self.op == '+':
            v = float('-inf')
            for (t1, t2) in self.m.composed_from[t]:
                dv = 0.
                if t1 is not None: dv += self.v1[t1]
                if t2 is not None: dv += self.v2[t2]
                v = logaddexp(v, dv)
        return v

    def __iter__(self): raise NotImplementedError()
    def __len__(self): raise NotImplementedError()

def compose(m1, m2):
    """Compose two finite transducers `m1` and `m2`, feeding the output of `m1`
    into the input of `m2`.

    In the resulting transducer `m`, for each transition `t`,
    `m.composed_from[t]` contains extra information about where `t`
    came from:

    - `(t1, t2)` means that `t` simulates `m1` following transition `t1` and
      `m2` following transition `t2`.
    - `(t1, None)` means that `t` simulates `m1` following transition `t1` and
      `m2` doing nothing.
    - `(None, t2)` means that `t` simulates m1 doing nothing and `m2`
      following transition `t2`.

    Instead of working with `m.composed_from` directly, it may be
    easier to use `m.compose_values`.
    """
    
    m = ComposedFST()
    m.composed_from = collections.defaultdict(list)
    m.set_start((m1.start, m2.start))

    m1_deletes = m2_inserts = False

    # Breadth-first search through the composed transducer
    agenda = collections.deque([(m1.start, m2.start)])
    index = set(agenda)

    def add_transition(t, t1, t2):
        m.add_transition(t)
        m.composed_from[t].append((t1, t2))
        r = t.r
        if r not in index:
            agenda.append(r)
            index.add(r)

    while len(agenda) > 0:
        q1, q2 = q = agenda.popleft()
        mtq1 = m1.transitions_inverse[q1]
        mtq2 = m2.transitions[q2]
        bs = min([mtq1, mtq2], key=len)
        for b in bs:
            if b == EPSILON: continue
            mtq2b = mtq2.get(b, [])
            for t1 in mtq1.get(b, []):
                for t2 in mtq2b:
                    t = Transition(q, t1.a, t2.b, (t1.r, t2.r))
                    add_transition(t, t1, t2)
            
        for t1 in mtq1.get(EPSILON, []):
            m1_Deletes = True
            t = Transition(q, t1.a, EPSILON, (t1.r, q2))
            add_transition(t, t1, None)

        for t2 in mtq2.get(EPSILON, []):
            m2_inserts = True
            t = Transition(q, EPSILON, t2.b, (q1, t2.r))
            add_transition(t, None, t2)
            
    m.set_accept((m1.accept, m2.accept))

    if m1_deletes and m2_inserts:
        raise NotImplementedError("I can't compose a deleting FST with an inserting FST")
    
    return m

def topological_sort(m):
    """Topologically sort the states of m. Returns a list `o` of states
    such that if `i < j`, then there is no path from `o[j]` to `o[i]`.
    If there is no such ordering, an exception is raised.
    """
    # https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
    order = []
    visited = set()
    path = set()
    def visit(q):
        if q in visited:
            return
        if q in path:
            raise ValueError("m must be acyclic")
        path.add(q)
        for a, ts in m.transitions[q].items():
            for t in ts:
                visit(t.r)
        path.remove(q)
        visited.add(q)
        order.append(q)
    visit(m.start)
    order.reverse()
    return order

    
