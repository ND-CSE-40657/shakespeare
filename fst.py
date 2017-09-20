import collections
import math

EPSILON="ε"
STOP="</s>"

class Transition(object):
    def __init__(self, q, a, r):
        self.q, self.a, self.r = q, a, r
    def __eq__(self, other):
        return (isinstance(other, Transition) and
                (self.q, self.a, self.r) == (other.q, other.a, other.r))
    def __ne__(self, other):
        return not self == other
    def __hash__(self):
        return hash((self.q, self.a, self.r))

class FST(object):
    def __init__(self):
        self.states = set()
        self.transitions_from = collections.defaultdict(lambda: collections.defaultdict(float))
        self.transitions_to = collections.defaultdict(lambda: collections.defaultdict(float))
        self.transitions_on = collections.defaultdict(lambda: collections.defaultdict(float))
        self.start = None
        self.accept = None

    def add_state(self, q):
        """Adds state q."""
        self.states.add(q)

    def set_start(self, q):
        """Sets the start state to q."""
        self.add_state(q)
        self.start = q

    def set_accept(self, q):
        """Sets the accept state to q."""
        self.add_state(q)
        self.accept = q

    def add_transition(self, t, wt=1):
        """Adds the transition 
             a:b/wt
           q ------> r
        
        If q and r are not already states, they are added too.
        If t is already a transition, its weight is incremented by wt."""
        self.add_state(t.q)
        self.add_state(t.r)
        self.transitions_from[t.q][t] += wt
        self.transitions_to[t.r][t] += wt
        self.transitions_on[t.a[0]][t] += wt

    def reweight_transition(self, t, wt=1):
        """Replaces the weight of transition t with new weight wt."""
        self.transitions_from[t.q][t] = wt
        self.transitions_to[t.r][t] = wt
        self.transitions_on[t.a[0]][t] = wt

    def train_joint(self, data):
        """Trains the transducer on the given data."""
        c = collections.Counter()
        for line in data:
            q = self.start
            for a in list(line) + [STOP]:
                for t in self.transitions_from[q]:
                    if a == t.a[0]:
                        c[t] += 1
                        q = t.r
                        break
                else:
                    raise ValueError("training string is not in language")
        for q in self.states:
            z = sum(self.transitions_from[q].values())
            for t in self.transitions_from[q]:
                self.reweight_transition(t, c[t]/z)

    def visualize(self):
        """Pops up a window showing a transition diagram.

        Requires graphviz.
        Under MacOS Sierra, you'll need to upgrade to 10.12.2 or newer."""
        
        import subprocess
        from tkinter import (Tk, Canvas, PhotoImage,
                             Scrollbar, HORIZONTAL, VERTICAL, X, Y,
                             BOTTOM, RIGHT, LEFT, YES, BOTH, ALL)
        import base64

        def escape(s):
            return '"{}"'.format(s.replace('\\', '\\\\').replace('"', '\\"'))
        
        dot = []
        dot.append("digraph {")
        dot.append('START[label="",shape=none];')
        index = {}
        for i, q in enumerate(self.states):
            index[q] = i
            attrs = {'label': escape(str(q)), 'fontname': 'Courier'}
            if q == self.accept:
                attrs['peripheries'] = 2
            dot.append('{} [{}];'.format(i, ",".join("{}={}".format(k, v) for (k, v) in attrs.items())))
        dot.append("START->{};".format(index[self.start]))
        for q in self.states:
            ts = collections.defaultdict(list)
            for t in self.transitions_from.get(q, []):
                ts[t.r].append(t)
            for r in ts:
                if len(ts[r]) > 8:
                    label = "\n".join(":".join(map(str, t.a)) for t in ts[r][:5]) + "\n..."
                else:
                    label = "\n".join(":".join(map(str, t.a)) for t in ts[r])
                dot.append('{}->{}[label={},fontname=Courier];'.format(index[q], index[r], escape(label)))
        dot.append("}")
        dot = "\n".join(dot)
        print(dot)
        proc = subprocess.run(["dot", "-T", "gif"], input=dot.encode("utf8"), stdout=subprocess.PIPE)
        if proc.returncode == 0:
            root = Tk()
            scrollx = Scrollbar(root, orient=HORIZONTAL)
            scrollx.pack(side=BOTTOM, fill=X)
            scrolly = Scrollbar(root, orient=VERTICAL)
            scrolly.pack(side=RIGHT, fill=Y)

            canvas = Canvas(root)
            image = PhotoImage(data=base64.b64encode(proc.stdout))
            canvas.create_image(0, 0, image=image, anchor="nw")
            canvas.pack(side=LEFT, expand=YES, fill=BOTH)
            
            canvas.config(xscrollcommand=scrollx.set, yscrollcommand=scrolly.set)
            canvas.config(scrollregion=canvas.bbox(ALL))
            scrollx.config(command=canvas.xview)
            scrolly.config(command=canvas.yview)

            root.mainloop()
        
def compose(m1, m2):
    """Compose two finite transducers m1 and m2, feeding the output of m1
    into the input of m2.

    In the resulting transducer, each transition t contains extra
    information about where it came from in the attribute
    t.composed_from:

    - (t1, t2) means that t simulates m1 following transition t1 and
      m2 following transition t2.
    - (t1, None) means that t simulates m1 following transition t1 and
      m2 doing nothing.
    - (None, t2) means that t simulates m1 doing nothing and m2
      following transition t2.
    """
    
    m = FST()
    m1_deletes = False
    m2_inserts = False

    m.set_start((m1.start, m2.start))
    for a in m1.transitions_on:
        for t1, wt1 in m1.transitions_on[a].items():
            if t1.a[1] != EPSILON:
                for t2, wt2 in m2.transitions_on.get(t1.a[1], {}).items():
                    t = Transition((t1.q, t2.q), (t1.a[0], t2.a[1]), (t1.r, t2.r))
                    t.composed_from = (t1, t2)
                    m.add_transition(t, wt=wt1*wt2)
            else:
                m1_deletes = True
                for q2 in m2.states:
                    t = Transition((t1.q, q2), (t1.a[0], EPSILON), (t1.r, q2))
                    t.composed_from = (t1, None)
                    m.add_transition(t, wt=wt1)
    for q1 in m1.states:
        for t2, wt2 in m2.transitions_on.get(EPSILON, {}).items():
            m2_inserts = True
            t = Transition((q1, t2.q), (EPSILON, t2.a[1]), (q1, t2.r))
            t.composed_from = (None, t2)
            m.add_transition(t, wt=wt2)
    m.set_accept((m1.accept, m2.accept))
    if m1_deletes and m2_inserts:
        raise ValueError("Can't compose a deleting FST with an inserting FST")
    return m

def make_ngram(data, n):
    """Create an n-gram language model from data. The data should be a
    list of lists of symbols."""
    m = FST()
    m.set_start(("<s>",) * (n-1))
    for line in data:
        q = m.start
        for a in line:
            q_next = q[1:] + (a,)
            m.add_transition(Transition(q, (a, a), q_next))
            q = q_next
        m.add_transition(Transition(q, (STOP, STOP), STOP))
    m.set_accept(STOP)
    return m
