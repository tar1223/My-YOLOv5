import platform


# 운영체제가 Windows인 경우에는 이모지를 제거하고, 그 외에는 그대로 반환
def emojis(str=''):
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str
