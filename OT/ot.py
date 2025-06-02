import torch
import numpy as np
import argparse
from diffusers import  DDIMScheduler

def asymmetry_index(A):
    """
    Calculate the asymmetry index of a matrix, ignoring diagonal elements
    :param A: Input PyTorch Tensor matrix
    :return: Asymmetry index, ranging from 0.0 to 1.0
    """
    # Clone the matrix to avoid modifying the original
    A = A.clone()
    # Set diagonal elements to 0
    diag_indices = torch.arange(min(A.size()))
    A[diag_indices, diag_indices] = 0
    # Compute the difference between the matrix and its transpose
    diff = A - A.T
    # Compute the Frobenius norm of the difference matrix
    diff_norm = torch.norm(diff, p='fro')
    # Compute the Frobenius norm of the processed matrix
    A_norm = torch.norm(A, p='fro')
    # Normalize the asymmetry index
    if A_norm == 0:
        return torch.tensor(0.0, dtype=A.dtype, device=A.device)
    index = diff_norm / (torch.sqrt(torch.tensor(2.0, dtype=A.dtype, device=A.device)) * A_norm)
    return index


def main():
    parser = argparse.ArgumentParser(description="sampling script for COCO14 on chongqing mach_ine.")
    parser.add_argument('--test_num', type=int, default=1)
    parser.add_argument('--scheduler', type=str, default='vp', choices=['vp', 'sub-vp', 'edm', 've'])
    parser.add_argument('--initial', type=str, default='non-affine', choices=['non-affine', 'affine'])
    parser.add_argument('--num_inference_steps', type=int, default=20)
    args = parser.parse_args()

    test_num = args.test_num
    scheduler = args.scheduler
    initial = args.initial
    num_inference_steps = args.num_inference_steps
    dtype = torch.float32

    sche = DDIMScheduler(beta_end=0.012, beta_start=0.00085, beta_schedule='scaled_linear', clip_sample=False, timestep_spacing='linspace', set_alpha_to_one=False)

    if initial in ['non-affine']:
        y_1 = torch.tensor([0.5, 0], dtype=torch.float32, device='cuda')
        y_2 = torch.tensor([0, -0.5], dtype=torch.float32, device='cuda')
        y_3 = torch.tensor([0., 0.], dtype=torch.float32, device='cuda')

    else:
        y_1 = torch.tensor([0, 0], dtype=torch.float32, device='cuda')
        y_2 = torch.tensor([0, 0.5], dtype=torch.float32, device='cuda')
        y_3 = torch.tensor([0, -0.6], dtype=torch.float32, device='cuda')


    sche.set_timesteps(num_inference_steps, device='cuda')
    timesteps = sche.timesteps
    a_rate = []
    for j in range(test_num):
        print('j = ', j)
        states = torch.randn(y_1.shape, generator=None, device='cuda', dtype=dtype)
        A = torch.eye(2).to('cuda')

        for i, t in enumerate(timesteps):
            if i < num_inference_steps - 1:
                alpha_s = sche.alphas_cumprod[timesteps[i + 1]].to(torch.float32)
                alpha_t = sche.alphas_cumprod[t].to(torch.float32)
                dt = t - timesteps[i + 1]
            else:
                break
                alpha_s = 1
                alpha_t = sche.alphas_cumprod[t].to(torch.float32)
                dt = t - 0

            dt =  - dt / 1000.0

            if scheduler in ['vp']:
                sigma_s = (1. - alpha_s) ** 0.5
                sigma_t = (1. - alpha_t) ** 0.5
                alpha_s = alpha_s ** 0.5
                alpha_t = alpha_t ** 0.5

                beta_t = sche.betas[t]
                f_t = -0.5*beta_t*1000
                g_t = torch.sqrt(beta_t*1000)
            if scheduler in ['sub-vp']:
                sigma_s = (1. - alpha_s) ** 0.5
                sigma_t = (1. - alpha_t) ** 0.5
                alpha_s = alpha_s ** 0.5
                alpha_t = alpha_t ** 0.5

                beta_t = sche.betas[t]
                beta_0 = sche.betas[-1]
                beta_1 = sche.betas[0]
                discount = 1. - torch.exp(-2 * beta_0*1000 * (t/1000.0) - (beta_1*1000 - beta_0*1000) * (t/1000.0) ** 2)
                f_t = -0.5*beta_t*1000
                g_t = torch.sqrt(beta_t*discount*1000)
            if scheduler in ['edm']:
                alpha_t = 1
                sigma_t = t/1000.0
                t_next = timesteps[i + 1]
                alpha_s = 1
                sigma_s = t_next/1000.0
                f_t = 0
                g_t = torch.sqrt(2*t/1000.0)
            if scheduler in ['ve']:
                alpha_t = 1
                sigma_t = torch.sqrt(t/1000.0)
                t_next = timesteps[i + 1]
                alpha_s = 1
                sigma_s = torch.sqrt(t_next/1000.0)
                f_t = 0
                g_t = 1

            v_1 = torch.exp(-torch.sum((states - alpha_t * y_1) ** 2) / (2 * sigma_t ** 2))
            v_2 = torch.exp(-torch.sum((states - alpha_t * y_2) ** 2) / (2 * sigma_t ** 2))
            v_3 = torch.exp(-torch.sum((states - alpha_t * y_3) ** 2) / (2 * sigma_t ** 2))

            # Calculate v_sum
            v_sum = v_1 + v_2 + v_3

            # Calculate w_i
            w_1 = v_1 / v_sum
            w_2 = v_2 / v_sum
            w_3 = v_3 / v_sum

            # Calculate y_theta
            y_theta = w_1 * y_1 + w_2 * y_2 + w_3 * y_3

            # Calculate outer-product
            op_1 = torch.outer(y_1, y_1)
            op_2 = torch.outer(y_2, y_2)
            op_3 = torch.outer(y_3, y_3)
            op_theta = torch.outer(y_theta, y_theta)

            # Define identity matrix
            I_2 = torch.eye(2).to('cuda')

            # Calculate B(t)
            B_t = (
                    (f_t - (g_t ** 2) / (2 * sigma_t ** 2)) * I_2 +
                    (alpha_t ** 2 * g_t ** 2) / (2 * sigma_t ** 4) * (
                            w_1 * op_1 + w_2 * op_2 + w_3 * op_3 - op_theta)
            )

            A = A + B_t @ A * dt

            # ddim update
            noise_pred = -(alpha_t * y_theta - states)/(sigma_t)
            coef_xt = alpha_s / alpha_t
            coef_eps = sigma_s - sigma_t * coef_xt
            states = coef_xt * states + coef_eps * noise_pred


        a_rate.append(asymmetry_index(A).item())


    print('average a_rate = ', np.mean(np.array(a_rate)))


if __name__ == '__main__':
    main()