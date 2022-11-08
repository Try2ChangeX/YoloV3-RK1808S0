import socket
import numpy as np
import threading
import cv2 as cv
from rknn.api import RKNN

class rknn_server:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.sock.bind(('0.0.0.0', port))

    def __del__(self):
        self.sock.close()

    def service(self, model, post_func):
        try:
            self.sock.listen(1)
            print('start listen...')

            while 1:
                conn, addr = self.sock.accept()
                t = threading.Thread(target=self.__deal, args=(conn, addr, model, post_func))
                t.start()

        except socket.error as msg:
            self.sock.close()
            print(msg)
            return -1

        return 0

    def __deal(self, conn, addr, model, post_func):
        print('connect from:'+str(addr))

        try:
            rknn = RKNN()
            ret = rknn.load_rknn(path=model)

            # init runtime environment
            print('--> Init runtime environment')
            ret = rknn.init_runtime()
            if ret != 0:
                print('Init runtime environment failed')
                exit(ret)
            print('done')
            conn.send(b'ready')

            while 1:
                decimg = self.__recieve_frame(conn)
                if decimg is None:
                    break

                outputs = rknn.inference(inputs=[decimg])
                data = post_func(outputs)
                self.__send_result(conn, data)
        except socket.error as msg:
            print(msg)

        conn.close()
        rknn.release()
        print("__deal finish")

    def __recieve_frame(self, conn):
        try :
            length = self.__recvall(conn, 16)
            stringData = self.__recvall(conn, int(length))
            data = np.frombuffer(stringData, np.uint8)
            decimg=cv.imdecode(data,cv.IMREAD_COLOR)
        except (RuntimeError, TypeError, NameError):
            return None

        return decimg

    def __recvall(self, conn, count):
        buf = b''
        while count:
            newbuf = conn.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def __pack_np(self, data):
        len_dt = np.dtype(np.int32)
        stringData = ""

        if data is None:
            type_name = "None".ljust(8)
            stringData = type_name.encode()
        else:
            type_name = str(data.dtype).ljust(8)
            shape_len = len(data.shape) * len_dt.itemsize
            shape = np.array(data.shape, dtype=len_dt)
            data_len = data.size*data.itemsize
            stringData = type_name.encode() + \
                         str.encode(str(shape_len).ljust(8)) + str.encode(str(data_len).ljust(8))
            stringData = stringData + shape.tostring() + data.tostring()

        return stringData
            
    def __send_result(self, conn, result):
        count = len(result)

        len_info = np.empty(count, dtype=np.int32)

        sock_data = b''

        for i in range(count):
            tmp = self.__pack_np(result[i])
            len_info[i] = len(tmp)
            sock_data = sock_data + tmp

        sock_data = str.encode(str(count).ljust(8)) + len_info.tostring() + sock_data

        conn.send(sock_data)

    def __mask2detect_results(self, mask):

        result_list = []
        retval, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
        bboxes = stats[stats[:, 4].argsort()][::-1]

        for b in bboxes:
            x0, y0 = b[0], b[1]
            x1 = b[0] + b[2]
            y1 = b[1] + b[3]

