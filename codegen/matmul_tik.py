from tbe import tik
from tbe.common.platform import set_current_compile_soc_info
import numpy as np
from te.platform.cce_conf import te_set_l2_mode
import sys

te_set_l2_mode(1)

def matmul_tik_compute(m_size, k_size, n_size):
    n_size = 16
    m_size = 16
    k_size = 16
    cube_len = 16
    C_loc_out_type = "float32"
    data_type = "float16"

    tik_instance = tik.Tik()
    dummy_arg = tik_instance.InputScalar(dtype="int32", name="dummy_arg")

    output_gm = tik_instance.Tensor(C_loc_out_type, (n_size//cube_len, m_size, cube_len), name="C_gm", scope=tik.scope_gm)
    matrix_a_gm = tik_instance.Tensor(data_type, [k_size//cube_len, m_size, cube_len], name="A_gm", scope=tik.scope_gm)
    matrix_b_gm = tik_instance.Tensor(data_type, [k_size//cube_len, n_size, cube_len], name="B_gm", scope=tik.scope_gm)
    matrix_a_l1 = tik_instance.Tensor(data_type, [k_size//cube_len, m_size, cube_len], name='l1_matrix_a', scope=tik.scope_cbuf)
    matrix_b_l1 = tik_instance.Tensor(data_type, [k_size//cube_len, n_size, cube_len], name='l1_matrix_a', scope=tik.scope_cbuf)
    output_l0c = tik_instance.Tensor(C_loc_out_type, [n_size//cube_len, m_size, cube_len], name='output_l0c', scope=tik.scope_cbuf_out)

    with tik_instance.for_range(0, 1, block_num = 1) as core_id:
        tik_instance.data_move(matrix_a_l1, matrix_a_gm, 0, k_size//cube_len, m_size, 0, 0)
        tik_instance.data_move(matrix_b_l1, matrix_b_gm, 0, k_size//cube_len, n_size, 0, 0)

        tik_instance.matmul(output_l0c, matrix_a_l1, matrix_b_l1, m_size, k_size, n_size, init_l1out=True)
        tik_instance.fixpipe(output_gm[:, :, :], output_l0c[:, :, :], n_size//cube_len, (m_size*cube_len*4)//32, 0, 0)
    tik_instance.BuildCCE(kernel_name='simple_matmul',
            inputs=[matrix_a_gm, matrix_b_gm], outputs=[output_gm], config={'l2_mode': 1, "tbe_debug_level": 2},
            flowtable=[dummy_arg], enable_l2=True, )
    return tik_instance

if __name__=='__main__':
    set_current_compile_soc_info("Ascend310")
    M, K, N = 16, 16, 16
    tik_instance = matmul_tik_compute(M, K, N)

    #data_x = np.ones((K//16, M, 16)).astype("float16")
    #data_y = np.ones((K//16, N, 16)).astype("float16")
    #data_y /= 2
    #feed_dict = {'A_gm': data_x, 'B_gm': data_y, 'dummy_arg':6}
    #model_data, = tik_instance.tikdb.start_debug(feed_dict=feed_dict,interactive=True)
    #print(model_data)
    #print(tik_instance)
