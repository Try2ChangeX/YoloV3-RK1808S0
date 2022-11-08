import socket
import numpy as np
import cv2 as cv

class rknn_client:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        try:
            self.sock.connect(("192.168.180.8", port))
            print('wait 1808 ready...')
            ready = self.sock.recv(5)
            print(ready)
            if ready != b'ready':
                return None

        except socket.error as msg:
            print(msg)
            return None

    def __del__(self):
        self.sock.close()

    def __send_frame(self, conn, frame):
        result, imgencode = cv.imencode('.jpg', frame)
        data = np.array(imgencode)
        stringData = data.tostring()
        sock_data = str.encode(str(len(stringData)).ljust(16)) + stringData
        conn.send(sock_data);

    def __recvall(self, conn, count):
        buf = b''
        while count:
            newbuf = conn.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def __unpake_np(self, string_data):
        len_dt = np.dtype(np.int32)
        type_name = string_data[0:7].decode().rstrip()
        if type_name == 'None':
            return None

        shape_len = int(string_data[8:15].decode().rstrip())
        shape_count = int(shape_len / len_dt.itemsize)

        data_dt = np.dtype(type_name)
        data_len = int(string_data[16:23].decode().rstrip())
        data_count = int(data_len / data_dt.itemsize)

        shape = np.frombuffer(string_data, len_dt, shape_count, offset = 24)
        data = np.frombuffer(string_data, data_dt, data_count, 24 + shape_len)
        data = data.reshape(shape)
        return data

    def __recieve_result(self, conn):
        len_dt = np.dtype(np.int32)

        stringData = self.__recvall(conn, 8)
        count = int(stringData.decode())

        stringData = self.__recvall(conn, count * len_dt.itemsize)
        #print("===stringData", stringData)
        len_info = np.frombuffer(stringData, len_dt)

        result = list()
        for i in range(len_info.size):
            stringData = self.__recvall(conn, len_info[i])
            data = self.__unpake_np(stringData)
            result.append(data)
    
        return result

    def inference(self, inputs):
        if len(inputs) != 1:
            return None

        img = inputs[0]
        self.__send_frame(self.sock, img)
        outputs = self.__recieve_result(self.sock)
        #print("===outputs", outputs)
        return outputs
