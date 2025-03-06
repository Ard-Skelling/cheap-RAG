import os
import base64
import hashlib
import hmac
import json
import datetime


# 加盐，对抗彩虹表攻击
SALT = os.getenv('SALT', 'aIie8i')

def base64url_encode(data):
    """
    将数据编码为 Base64 URL 安全格式
    """
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')

def base64url_decode(data):
    """
    将 Base64 URL 安全格式的数据解码
    """
    padding = b'=' * (4 - (len(data) % 4))
    return base64.urlsafe_b64decode(data.encode('utf-8') + padding)

def generate_jwt(payload, secret_key, salt, algorithm="HS256"):
    """
    生成 JWT
    :param payload: 用户id和过期时间（字典）
    :param secret_key: 原始密钥
    :param salt: 盐值
    :param algorithm: 签名算法（目前仅支持 HS256）
    :return: JWT 字符串
    """
    # 结合密钥和盐值生成最终的签名密钥
    signing_key = f"{secret_key}{salt}".encode('utf-8')

    header = {
        "alg": algorithm,
        "typ": "JWT"
    }
    header_encoded = base64url_encode(json.dumps(header).encode('utf-8'))

    payload_encoded = base64url_encode(json.dumps(payload).encode('utf-8'))

    signature_input = f"{header_encoded}.{payload_encoded}".encode('utf-8')
    signature = hmac.new(signing_key, signature_input, hashlib.sha256).digest()
    signature_encoded = base64url_encode(signature)

    jwt_token = f"{header_encoded}.{payload_encoded}.{signature_encoded}"
    return jwt_token

def verify_jwt(jwt_token, secret_key, salt):
    """
    验证 JWT
    :param jwt_token: JWT 字符串
    :param secret_key: 原始密钥
    :param salt: 盐值
    :return: 用户id和过期时间（字典）或 None（如果验证失败）
    """
    try:
        # 结合密钥和盐值生成最终的签名密钥
        signing_key = f"{secret_key}{salt}".encode('utf-8')

        # 拆分 JWT
        header_encoded, payload_encoded, signature_encoded = jwt_token.split('.')

        # 重新计算签名
        signature_input = f"{header_encoded}.{payload_encoded}".encode('utf-8')
        expected_signature = hmac.new(signing_key, signature_input, hashlib.sha256).digest()
        expected_signature_encoded = base64url_encode(expected_signature)

        # 验证签名
        if not hmac.compare_digest(signature_encoded, expected_signature_encoded):
            return None
        
        # 解码 Payload
        payload = json.loads(base64url_decode(payload_encoded))

        # 检查过期时间
        if "exp" in payload:
            expiration_time = datetime.datetime.fromtimestamp(payload["exp"])
            if datetime.datetime.now(datetime.timezone.utc) > expiration_time:
                return None

        return payload
    except Exception as e:
        print(f"验证 JWT 失败: {e}")
        return None


if __name__ == "__main__":
    # 原始密钥
    secret_key = "your_secret_key"

    # 生成 JWT
    payload = {
        "user_id": "12345",
        "exp": (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3600)).timestamp()
    }
    jwt_token = generate_jwt(payload, secret_key, SALT)
    print(f"生成的 JWT: {jwt_token}")

    # 验证 JWT
    verified_payload = verify_jwt(jwt_token, secret_key, SALT)
    if verified_payload:
        print(f"验证成功，Payload: {verified_payload}")
    else:
        print("验证失败")