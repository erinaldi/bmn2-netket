import netket as nk
from netket.operator.spin import sigmax,sigmaz 
from scipy.sparse.linalg import eigsh
import jax.numpy as jnp
import flax.linen as nn
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

class MF(nn.Module):
    @nn.compact
    def __call__(self, x):
        lam = self.param(
            "lambda", nn.initializers.normal(), (1,), float
        )
        
        # compute the probabilities
        p = nn.log_sigmoid(lam*x)
        # sum the output
        return 0.5 * jnp.sum(p, axis=-1)


def main():
    N = 20
    hi = nk.hilbert.Spin(s=1 / 2, N=N)
    Gamma = -1
    H = sum([Gamma*sigmax(hi,i) for i in range(N)])
    V=-1
    H += sum([V*sigmaz(hi,i)*sigmaz(hi,(i+1)%N) for i in range(N)])

    # sparse exact hamiltonian matrix elements
    # sp_h=H.to_sparse()
    # eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
    # if nk.utils.mpi.rank == 0:
    #     print("eigenvalues with scipy sparse:", eig_vals)

    # Create an instance of the model. 
    mf_model = MF()
    # Create the local sampler on the hilbert space
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=4)
    # Construct the variational state using the model and the sampler above.
    vstate = nk.vqs.MCState(sampler, mf_model, n_samples=2**10)
    # optimizer
    optimizer = nk.optimizer.Sgd(learning_rate=0.05)
    # build the optimisation driver
    gs = nk.driver.VMC(H, optimizer, variational_state=vstate)
    # run the driver for 300 iterations. This will display a progress bar
    gs.run(n_iter=300)
    mf_energy=vstate.expect(H)
    # error=abs((mf_energy.mean-eig_vals[0])/eig_vals[0])
    if nk.utils.mpi.rank == 0:
        # print("Optimized energy and relative error (compared to sparse ED): ",mf_energy,error)
        print("Optimized energy with mean field ansatz: ",mf_energy)


if __name__ == "__main__":
    # run this like this: mpirun -np 4 python benchmarks/ising1d.py
    # without MPI: 38 it/s from the progress bar (45-50 it/s with n_chains=4)
    # with MPI np=2: 48 it/s (66 it/s with n_chains=4)
    # with MPI np=4: 65 it/s (120 it/s with n_chains=4)
    main()