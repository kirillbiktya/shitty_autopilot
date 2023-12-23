import socket
import struct
import time
import threading


class Controller:
    """
    Интерфейс для управления автомобилем и считывания основных его характеристик
    (скорость, обороты, положение рулевых колес)
    """
    def __init__(self, client_name: str, security_code: int, local_host='192.168.1.5', remote_host='192.168.1.8'):
        """

        :param client_name: Название клиентской программы
        :param security_code: Код безопасности (может быть извлечен из QR-кода в игре
        :param local_host: адрес компьютера клиента
        :param remote_host: адрес компьютера с запущенной на нем игрой
        """
        self.throttle = 0.0  # степень нажатия педали газа
        self.steering = 0.5  # положение рулевого колеса (0 - влево, 0.5 - центр, 1 - вправо)
        self.brakes = 0.0  # степень нажатия педали тормоза

        self.rpm = 0  # об./мин
        self.speed = 0.  # км/ч

        self._recv_active = True
        self._send_active = True

        self._recv_thread = None
        self._send_thread = None

        self.client_name = client_name
        self.security_code = security_code
        self.local_host = local_host
        self.remote_host = remote_host

        self.recv_port = 4445
        self.send_port = 4444

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._socket.settimeout(0.1)
        self._socket.bind((self.local_host, self.recv_port))

        self._do_pair()

    def __del__(self):
        self._recv_active = False
        self._send_active = False
        try:
            self._recv_thread.join()
            self._send_thread.join()
        except Exception as e:
            print(e)

        self._socket.close()

    def _do_pair(self):
        """
        Отреверсенный механизм аутентификации клиента, https://github.com/BeamNG/remotecontrol
        :return:
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)  # UDP
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind((self.local_host, 0))
        sock.sendto(
            ('beamng|' + self.client_name + '|' + str(self.security_code)).encode('utf-8'),
            ("255.255.255.255", self.send_port)
        )
        sock.close()
        success = False

        while not success:
            try:
                data = self._socket.recv(256)
                if data.decode("utf-8") == 'beamng|' + str(self.security_code):
                    print("Paired successfully")
                    success = True
            except socket.timeout:
                print("Timeout error. Maybe wrong security code?")
                break

    def _recv_cycle(self):
        """
        В цикле получаем данные от игры, распаковываем структуру с данными и сохраняем нужное
        :return:
        """
        while self._recv_active:
            try:
                data = self._socket.recv(256)
                if not data:
                    continue

                outgauge_pack = struct.unpack('<I3sxH2B7f2I3f15sx15sxi', data)
                self.speed = int(outgauge_pack[5]) * 3.6
                self.rpm = int(outgauge_pack[6])
            except socket.timeout:
                continue

    def _send_cycle(self):
        """
        Упаковываем данные о нажатых педалях, положении рулевого колеса и отправляем в игру
        :return:
        """
        while self._send_active:
            time.sleep(0.06)
            try:
                data = struct.pack('>4f', *[self.steering, self.throttle, self.brakes, round(time.time() * 1000)])
                self._socket.sendto(data, (self.remote_host, self.send_port))
            except Exception as e:
                print(e)
                continue

    def run(self):
        """
        Запускаем циклы получения и отправки данных в отдельных потоках
        :return:
        """
        self._recv_thread = threading.Thread(target=self._recv_cycle)
        self._send_thread = threading.Thread(target=self._send_cycle)
        self._recv_thread.start()
        self._send_thread.start()
