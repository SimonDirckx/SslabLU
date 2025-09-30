import argparse
import pickle
import matplotlib.pyplot as plt

def load_svd_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    # Adjust the key to match your data structure!
    return data['svd_Scl']


####
plt.figure()


H = 1; p = 8
for ainv in [2,4,8,16]:
    pickle_loc = "thinslab_H%2.5f_p%d_ainv%d.pkl" % (H,p,ainv)

    svd_data = load_svd_from_pickle(pickle_loc)
    plt.semilogy(svd_data,label='a=%5.5f'%(1/ainv))

plt.xlabel("Index")
plt.ylabel("Singular Value")
plt.title(
    rf"Singular Value Spectrum of $S_{{j,j-1}}$ with Slab Width $H={H:.2f}$" + "\n"
    r"HPS with $p=8$ for Various $a$"
)
plt.xlim(-100, 1100)

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(title="Patch size $2a \\times 2a \\times 2a$",fontsize=10,title_fontsize=12)
plt.tight_layout()
plt.savefig("glr_hps_withtorch_H1.png")
plt.show()

plt.figure()
H = 0.25; p = 8
for ainv in [8,16,32]:
    pickle_loc = "thinslab_H%2.5f_p%d_ainv%d.pkl" % (H,p,ainv)

    svd_data = load_svd_from_pickle(pickle_loc)
    plt.semilogy(svd_data,label='a=%5.5f'%(1/ainv))

plt.xlabel("Index")
plt.ylabel("Singular Value")
plt.title(
    rf"Singular Value Spectrum of $S_{{j,j-1}}$ with Slab Width $H={H:.2f}$" + "\n"
    r"HPS with $p=8$ for Various $a$"
)

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(title="Patch size $2a \\times 2a \\times 2a$",fontsize=10,title_fontsize=12)
plt.tight_layout()
plt.savefig("glr_hps_withtorch_H025.png")
plt.show()

plt.figure()

ainv = 64
p = 2
for nslabs in [1, 2,4,8]:
    H = 1.0 / nslabs
    pickle_loc = f"thinslab_H{H:.5f}_p{p}_ainv{ainv}.pkl"
    svd_data = load_svd_from_pickle(pickle_loc)
    plt.semilogy(svd_data, label=f'H={H:.5f}')

plt.xlabel("Index")
plt.ylabel("Singular Value")
plt.title(
    fr"Singular Value Spectrum of $S_{{j,j-1}}$ with Varying Slab Width $H$" + "\n"
    fr"FD discretization with 1/h={1.0/ainv:5.2e}"
)


plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(title="Varying Slab Width $H$",fontsize=10,title_fontsize=12)
plt.tight_layout()
plt.show()