import numpy as np, itertools
from scipy.linalg import expm
rng=np.random.default_rng(0)
I=np.eye(2);X=np.array([[0,1],[1,0]]);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1.,-1])
def kron(*a):
    r=np.array([[1.]])
    for m in a: r=np.kron(r,m)
    return r
n=3;H=kron(Z,Z,I)+kron(I,Z,Z)
paulis=[kron(*p) for p in itertools.product([I,X,Y,Z],repeat=n)][1:]
E=np.linalg.eigvalsh(H); ell=2*(E.max()-E.min()); eta=0.5/ell
def run(psi,T=20000,uniform=False):
    for t in range(T):
        if uniform:
            A=rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)); s=(A+A.conj().T)/2
            s-=np.trace(s)/8*np.eye(8); s/=np.linalg.norm(s)
        else: s=paulis[rng.integers(len(paulis))]
        d=np.real(1j*psi.conj()@(s@H-H@s)@psi)   # dE/dtheta for exp(-i theta s)
        psi=expm(-1j*(-eta*d)*s)@psi
    return np.real(psi.conj()@H@psi)
psi0=np.zeros(8,complex);psi0[0]=np.cos(0.4);psi0[1]=np.sin(0.4)
print("E0",E.min(),"trap Pauli:",run(psi0.copy()))
p=psi0.copy();p[2]+=1e-6;p/=np.linalg.norm(p);print("perturbed Pauli:",run(p))
print("trap uniform dirs:",run(psi0.copy(),T=4000,uniform=True))
def run2(psi,T=5000):
    for t in range(T):
        s=paulis[rng.integers(len(paulis))]
        d=np.real(1j*psi.conj()@(s@H-H@s)@psi)
        psi=expm(1j*eta*d*s)@psi
    return psi
for trial in range(3):
    a=rng.normal(size=2)+1j*rng.normal(size=2); a/=np.linalg.norm(a)
    p=np.zeros(8,complex); p[0],p[1]=a
    g=2*(H-np.real(p.conj()@H@p)*np.eye(8))@p
    q=run2(p)
    print("init E",round(float(np.real(p.conj()@H@p)),3),"|g|",round(float(np.linalg.norm(g)),3),
          "final E",round(float(np.real(q.conj()@H@q)),6),"weights",np.round(np.abs(q)**2,4))
