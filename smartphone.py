import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, savgol_filter
import numpy as np
import pandas as pd
import streamlit as st

# Parâmetros do filtro e da taxa de amostragem
target_fs = 100  # Frequência de amostragem desejada (100 Hz)
butter_cutoff_lowpass = 10  # Frequência de corte do filtro passa-baixa (10 Hz)
# Frequência de corte do filtro passa-alta (0.5 Hz)
butter_cutoff_highpass = 0.5
savgol_window = 5  # Tamanho da janela para Savitzky-Golay
savgol_polyorder = 3  # Ordem do polinômio para Savitzky-Golay

# Função para aplicar filtro Butterworth passa-baixa


def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Função para aplicar filtro Butterworth passa-alta


def butter_highpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)


st.set_page_config(layout="wide")

# Carregar o arquivo de texto
st.title("Processamento de Dados - Projeto Suellen")
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader(
        "Faça o upload de um arquivo de texto com cabeçalho e quatro colunas numéricas", type="csv")

    if uploaded_file is not None:
        print('2')
        # Ler o arquivo de texto, assumindo que a primeira linha é o cabeçalho
        data = pd.read_csv(uploaded_file, delimiter=";")

        # Verificar se o arquivo possui exatamente 4 colunas numéricas
        if len(data.columns) == 4 and data.select_dtypes(include=['number']).shape[1] == 4:
            # Separar as colunas em variáveis de tempo e acelerações
            t = data.iloc[:, 0].values  # Tempo
            t = t/1000
            acc_x = data.iloc[:, 1].values  # Aceleração em X
            acc_y = data.iloc[:, 2].values  # Aceleração em Y
            acc_z = data.iloc[:, 3].values  # Aceleração em Z
            print('3')
            # Interpolação das séries temporais para a nova frequência de 100 Hz
            interp_func_x = interp1d(
                t, acc_x, kind='linear', fill_value="extrapolate")
            interp_func_y = interp1d(
                t, acc_y, kind='linear', fill_value="extrapolate")
            interp_func_z = interp1d(
                t, acc_z, kind='linear', fill_value="extrapolate")

            # Criação de nova linha temporal
            t_new = np.arange(t[0], t[-1], 1 / target_fs)
            acc_x_interp = interp_func_x(t_new)
            acc_y_interp = interp_func_y(t_new)
            acc_z_interp = interp_func_z(t_new)

            t_new = t_new

            # Definir início da visualização após 10 segundos
            maior_valor = np.max(acc_x_interp)
            for index, valor in enumerate(acc_x_interp):
                if valor == maior_valor:
                    start = index + 1000
                    break

            # Aplicação do filtro passa-alta para remover drift de baixa frequência
            acc_x_detrended = butter_highpass_filter(
                acc_x_interp, butter_cutoff_highpass, target_fs)
            acc_y_detrended = butter_highpass_filter(
                acc_y_interp, butter_cutoff_highpass, target_fs)
            acc_z_detrended = butter_highpass_filter(
                acc_z_interp, butter_cutoff_highpass, target_fs)

            # Aplicação do filtro Butterworth passa-baixa em 10 Hz
            acc_x_filtered = butter_lowpass_filter(
                acc_x_detrended, butter_cutoff_lowpass, target_fs)
            acc_y_filtered = butter_lowpass_filter(
                acc_y_detrended, butter_cutoff_lowpass, target_fs)
            acc_z_filtered = butter_lowpass_filter(
                acc_z_detrended, butter_cutoff_lowpass, target_fs)

            # Aplicação do filtro Savitzky-Golay para suavizar ainda mais os dados filtrados
            acc_x_smooth = savgol_filter(
                acc_x_filtered, window_length=savgol_window, polyorder=savgol_polyorder)
            acc_y_smooth = savgol_filter(
                acc_y_filtered, window_length=savgol_window, polyorder=savgol_polyorder)
            acc_z_smooth = savgol_filter(
                acc_z_filtered, window_length=savgol_window, polyorder=savgol_polyorder)

            # Cálculo da variação da velocidade em função do tempo (integral da aceleração)
            vel_x = np.cumsum(acc_x_smooth) / target_fs
            vel_y = np.cumsum(acc_y_smooth) / target_fs
            vel_z = np.cumsum(acc_z_smooth) / target_fs
            print('4')
            # Remover tendência linear (drift) da velocidade
            vel_x -= np.polyval(np.polyfit(t_new, vel_x, 1), t_new)
            vel_y -= np.polyval(np.polyfit(t_new, vel_y, 1), t_new)
            vel_z -= np.polyval(np.polyfit(t_new, vel_z, 1), t_new)

            # Cálculo do deslocamento em função do tempo (integral da velocidade)
            disp_x = np.cumsum(vel_x) / target_fs
            disp_y = np.cumsum(vel_y) / target_fs
            disp_z = np.cumsum(vel_z) / target_fs

            print('5')
            # Plotagem dos dados de deslocamento suavizados
            fig = plt.figure()
            plt.plot(t_new[start:], disp_x[start:],
                     'r', label='Deslocamento em X')
            plt.plot(t_new[start:], disp_z[start:],
                     'b', label='Deslocamento em Z')
            plt.xlabel("Tempo (s)")
            plt.ylabel("Grandeza")
            plt.legend()
            st.pyplot(fig)

            print('5.1')
            fig = plt.figure()
            plt.plot(t_new[start:], vel_x[start:],
                     'r', label='Velocidade em X')
            plt.plot(t_new[start:], vel_z[start:],
                     'b', label='Velocidade em Z')
            plt.xlabel("Tempo (s)")
            plt.ylabel("Grandeza")
            plt.legend()
            st.pyplot(fig)

            print('5.2')
            fig = plt.figure()
            plt.plot(t_new[start:], acc_x_smooth[start:],
                     'r', label='Aceleração em X')
            plt.plot(t_new[start:], acc_z_smooth[start:],
                     'b', label='Aceleração em Z')
            plt.xlabel("Tempo (s)")
            plt.ylabel("Grandeza")
            plt.legend()
            st.pyplot(fig)
            print('6')

            # Cálculo adicional de métricas
            rms_x_mobile = np.sqrt(np.mean(disp_x[start:3000] ** 2))
            rms_z_mobile = np.sqrt(np.mean(disp_z[start:3000] ** 2))
            deviation_disp_mobile = np.sum(
                np.sqrt(disp_x[start:] ** 2 + disp_z[start:] ** 2))

            rms_x_vel_mobile = np.sqrt(np.mean(vel_x[start:3000] ** 2))
            rms_z_vel_mobile = np.sqrt(np.mean(vel_z[start:3000] ** 2))
            mean_vel_x_mobile = np.mean(vel_x[start:3000])
            mean_vel_z_mobile = np.mean(vel_z[start:3000])
            deviation_vel_mobile = np.sum(
                np.sqrt(vel_x[start:] ** 2 + vel_z[start:] ** 2))

            rms_x_acc_mobile = np.sqrt(np.mean(acc_x[start:3000] ** 2))
            rms_z_acc_mobile = np.sqrt(np.mean(acc_z[start:3000] ** 2))
            mean_acc_x_mobile = np.mean(acc_x[start:3000])
            mean_acc_z_mobile = np.mean(acc_z[start:3000])
            deviation_acc_mobile = np.sum(
                np.sqrt(acc_x[start:] ** 2 + acc_z[start:] ** 2))

            st.write(f"RMS ML: {rms_x_mobile}")
            st.write(f"RMS AP: {rms_z_mobile}")
            st.write(f"Desvio Total: {deviation_disp_mobile}")
            st.write(f"RMS velocidade ML: {rms_x_vel_mobile}")
            st.write(f"RMS velocidade AP: {rms_z_vel_mobile}")
            st.write(f"Velocidade Média ML: {mean_vel_x_mobile}")
            st.write(f"Velocidade Média AP: {mean_vel_z_mobile}")
            st.write(f"Velocidade Média norma: {deviation_vel_mobile}")
            st.write(f"Aceleração Média ML: {mean_acc_x_mobile}")
            st.write(f"Aceleração Média AP: {mean_acc_z_mobile}")
            st.write(f"Aceleração Média norma: {deviation_acc_mobile}")
            st.write(f"RMS Aceleração ML: {rms_x_acc_mobile}")
            st.write(f"RMS Aceleraçao AP: {rms_z_acc_mobile}")

        else:
            st.error("O arquivo deve conter exatamente 4 colunas numéricas.")
    else:
        st.info("Por favor, faça o upload de um arquivo de texto.")
with col2:
    arquivo = st.file_uploader("Faça o upload do arquivo de texto", type=["txt", "csv"])

if arquivo is not None:
    try:
        # Carregar o arquivo com pandas
        # Ajuste `sep` conforme necessário (ex.: '\t' para tabulação, ',' para vírgulas, etc.)
        dados = pd.read_csv(arquivo, sep='\t')

        # Verificar se há pelo menos duas colunas no arquivo
        if dados.shape[1] >= 2:
            # Atribuir colunas a variáveis separadas
            x = dados.iloc[:, 0]  # Primeira coluna
            y = dados.iloc[:, 1]  # Segunda coluna
            # Configurar parâmetros
            fs = 30  # Taxa de amostragem em Hz
            dt = 1 / fs  # Intervalo de tempo entre amostras

            # Calcular o vetor temporal
            t = np.arange(0, len(x) * dt, dt)

            interp_func_x = interp1d(
                t, x, kind='linear', fill_value="extrapolate")
            interp_func_y = interp1d(
                t, y, kind='linear', fill_value="extrapolate")

            t_new = np.arange(t[0], t[-1], 1 / target_fs)
            disp_x_interp = interp_func_x(t_new)
            disp_y_interp = interp_func_y(t_new)

            # Definir início da visualização após 10 segundos
            for index, tempo in enumerate(t_new):
                if tempo > 10:
                    start = index
                    break

            # Aplicar o filtro Savitzky-Golay nos dados de deslocamento
            x_smooth = savgol_filter(
                disp_x_interp, window_length=5, polyorder=3)
            y_smooth = savgol_filter(
                disp_y_interp, window_length=5, polyorder=3)

            # Calcular o vetor de velocidade (usando diferenças finitas para derivada)
            vel_x = np.diff(x_smooth) / dt
            vel_y = np.diff(y_smooth) / dt

            # Adicionar uma última amostra ao vetor de velocidade para igualar o comprimento com o vetor de deslocamento
            vel_x = np.append(vel_x, vel_x[-1])
            vel_y = np.append(vel_y, vel_y[-1])
            vel_norm = np.sum(np.sqrt(vel_x[start:] ** 2 + vel_y[start:] ** 2))

            # Calcular o vetor de aceleração (derivada da velocidade)
            acc_x = np.diff(vel_x) / dt
            acc_y = np.diff(vel_y) / dt

            # Adicionar uma última amostra ao vetor de aceleração para igualar o comprimento com o vetor de velocidade
            acc_x = np.append(acc_x, acc_x[-1])
            acc_y = np.append(acc_y, acc_y[-1])
            acc_norm = np.sum(np.sqrt(acc_x[start:] ** 2 + acc_y[start:] ** 2))

            # Plotar dados de deslocamento suavizados
            fig = plt.figure()
            plt.plot(t_new[start:], x_smooth[start:], 'r',
                     label='Deslocamento X (Suavizado)')
            plt.plot(t_new[start:], y_smooth[start:], 'g',
                     label='Deslocamento Y (Suavizado)')
            plt.legend()

            st.pyplot(fig)

            # Plotar dados de velocidade
            fig = plt.figure()
            plt.plot(t_new[start:], vel_x[start:], 'r', label='Velocidade X')
            plt.plot(t_new[start:], vel_y[start:], 'g', label='Velocidade Y')
            plt.legend()

            st.pyplot(fig)

            # Plotar dados de aceleração
            fig = plt.figure()
            plt.plot(t_new[start:], acc_x[start:], 'r', label='Aceleração X')
            plt.plot(t_new[start:], acc_y[start:], 'g', label='Aceleração Y')
            plt.legend()

            st.pyplot(fig)

            # Cálculo adicional de métricas
            rms_x = np.sqrt(np.mean(x_smooth[start:3000] ** 2))
            rms_y = np.sqrt(np.mean(y_smooth[start:3000] ** 2))
            deviation = np.sum(
                np.sqrt(x_smooth[start:] ** 2 + y_smooth[start:] ** 2))

            rms_vel_x = np.sqrt(np.mean(vel_x[start:3000] ** 2))
            rms_vel_y = np.sqrt(np.mean(vel_y[start:3000] ** 2))
            mean_vel_x = np.mean(vel_x[start:3000])
            mean_vel_y = np.mean(vel_y[start:3000])
            deviation_vel = np.sum(vel_norm)

            rms_acc_x = np.sqrt(np.mean(acc_x[start:3000] ** 2))
            rms_acc_y = np.sqrt(np.mean(acc_y[start:3000] ** 2))
            mean_acc_x = np.mean(acc_x[start:3000])
            mean_acc_y = np.mean(acc_y[start:3000])
            deviation_acc = np.sum(acc_norm)

            st.write(f"RMS ML: {rms_x}")
            st.write(f"RMS AP: {rms_y}")
            st.write(f"Desvio Total: {deviation}")
            st.write(f"RMS velocidade ML: {rms_vel_x}")
            st.write(f"RMS velocidade AP: {rms_vel_y}")
            st.write(f"Velocidade Média ML: {mean_vel_x}")
            st.write(f"Velocidade Média AP: {mean_vel_y}")
            st.write(f"Velocidade Média norma: {deviation_vel}")
            st.write(f"Aceleração Média ML: {mean_acc_x}")
            st.write(f"Aceleração Média AP: {mean_acc_y}")
            st.write(f"Aceleração Média norma: {deviation_acc}")
            st.write(f"RMS Aceleração ML: {rms_acc_x}")
            st.write(f"RMS Aceleraçao AP: {rms_acc_y}")
        else:
            st.error("O arquivo deve conter exatamente 2 colunas numéricas.")
    else:
        st.info("Por favor, faça o upload de um arquivo CSV.")
