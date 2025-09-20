import math

def decode_hif8(byte_val: int):
    """
    Decodes an 8-bit integer into its HiFloat8 value and constituent fields.
    The decoding logic is based on the specifications in "Ascend HiFloat8 Format for Deep Learning".
    """
    if not 0 <= byte_val <= 255:
        raise ValueError("Input must be an 8-bit integer (0-255).")

    bits = f'{byte_val:08b}'
    sign_bit = bits[0]
    sign_val = -1.0 if sign_bit == '1' else 1.0
    params = {}

    # --- Handle Special Values (Table 4 & surrounding text) ---
    # Zero: bit-pattern 00000000 [cite: 101, 103]
    if byte_val == 0b00000000:
        params = {'Type': 'Zero', 'S': sign_bit, 'Dot': '0000', 'M': '000'}
        return (bits, 0.0, params)
    # NaN: bit-pattern 10000000 [cite: 101, 103]
    if byte_val == 0b10000000:
        params = {'Type': 'NaN', 'S': sign_bit, 'Dot': '0000', 'M': '000'}
        return (bits, math.nan, params)
    # Infinity: "2 bit-patterns with the largest absolute value" [cite: 96]
    # This corresponds to E=+15 and M=1 (for D=4), which are 0b01101111 and 0b11101111
    if byte_val == 0b01101111:
        params = {'Type': 'Infinity', 'S': '0', 'Dot': '11', 'Em': '0111', 'M': '1'}
        return (bits, math.inf, params)
    if byte_val == 0b11101111:
        params = {'Type': 'Infinity', 'S': '1', 'Dot': '11', 'Em': '0111', 'M': '1'}
        return (bits, -math.inf, params)

    # --- Parse Dot Field using Unconventional Prefix Codes (Table 2) ---
    dot_code = None
    em_bits = ''
    m_bits = ''
    
    # Case D=4 (Dot='11') [cite: 76, 77, 92]
    if bits[1:3] == '11':
        dot_code, D, m_width = '11', 4, 1
        em_bits, m_bits = bits[3:7], bits[7:]
    # Case D=3 (Dot='10') [cite: 76, 77, 92]
    elif bits[1:3] == '10':
        dot_code, D, m_width = '10', 3, 2
        em_bits, m_bits = bits[3:6], bits[6:]
    # Case D=2 (Dot='01') [cite: 76, 77, 92]
    elif bits[1:3] == '01':
        dot_code, D, m_width = '01', 2, 3
        em_bits, m_bits = bits[3:5], bits[5:]
    # Case D=1 (Dot='001') [cite: 76, 77, 92]
    elif bits[1:4] == '001':
        dot_code, D, m_width = '001', 1, 3
        em_bits, m_bits = bits[4:5], bits[5:]
    # Case D=0 (Dot='0001') [cite: 76, 77, 92]
    elif bits[1:5] == '0001':
        dot_code, D, m_width = '0001', 0, 3
        m_bits = bits[5:]
    # Case Denormal (Dot='0000') [cite: 76, 77, 92]
    elif bits[1:5] == '0000':
        m_bits = bits[5:]
        m_val = int(m_bits, 2)
        # Denormal formula: X = (-1)^S * 2^(M-23) * 1.0 
        exponent = m_val - 23
        value = sign_val * (2.0**exponent)
        params = {'Type': 'Denormal', 'S': sign_bit, 'Dot': '0000', 'M': m_bits}
        return (bits, value, params)
    else:
        # Should be unreachable
        return (bits, "Invalid Encoding", {})

    # --- Calculate Normal (NML) Value ---
    # Formula: X = (-1)^S * 2^E * 1.M 
    
    # Decode exponent from sign-magnitude Em with implicit leading '1' [cite: 81, 152]
    exponent_val = 0
    if D > 0:
        exp_sign_bit = em_bits[0]
        mag_suffix = em_bits[1:]
        mag_bin = '1' + mag_suffix
        mag_val = int(mag_bin, 2)
        exponent_val = mag_val if exp_sign_bit == '0' else -mag_val
    
    # Decode mantissa M
    m_val = int(m_bits, 2)
    significand = 1.0 + m_val / (2**m_width)
    
    value = sign_val * (2.0**exponent_val) * significand
    
    params = {
        'Type': 'Normal', 'S': sign_bit, 'Dot': dot_code, 
        'D': D, 'Em': em_bits, 'E': exponent_val, 'M': m_bits
    }
    return (bits, value, params)

import matplotlib.pyplot as plt
def scatter_hif8(values: list[float], title: str):
    """
    Scatter plot of HiFloat8 values.
    """
    plt.scatter(values, [0]*len(values), c='blue', alpha=0.7, s=30)
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel('')
    plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.text(0.02, 0.98, f'Total values: {len(values)}', transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    plt.savefig(title + '.png')
    plt.close()
    plt.show()

if __name__ == "__main__":
    values = []
    for i in range(256):
        bits, value, params = decode_hif8(i)
        values.append(value)
        params_str = ", ".join([f"{k}: {v}" for k, v in params.items()])
        print(f"{i:<6} {f'0x{i:02x}':<5} {bits:<12} {value:<25.10g} {params_str}")

    scatter_hif8(values, 'HiFloat8 Scatter Plot')
    print(f"Saved plot to HiFloat8 Scatter Plot.png")


        
        