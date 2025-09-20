from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Iterator, Union
from dataclasses import dataclass
import math


@dataclass
class NumericValue:
    """Represents a single numeric value with its code and properties."""
    code: int
    value: float
    code_binary: str
    category: str  # 'normal', 'subnormal', 'zero', 'infinity', 'nan', etc.


class NumericFormat(ABC):
    """Abstract base class for all numeric formats."""
    
    def __init__(self, name: str, total_bits: int):
        self.name = name
        self.total_bits = total_bits
        self._min_value = None
        self._max_value = None
    
    def _normalize_code_input(self, code_input: Union[int, str]) -> int:
        """
        Normalize various code input formats to an integer.
        
        Accepts:
        - Integer (including binary literals like 0b00001111)
        - Binary string (like "000011" or "0b000011")
        """
        if isinstance(code_input, int):
            return code_input
        elif isinstance(code_input, str):
            # Remove '0b' prefix if present
            binary_str = code_input.lstrip('0b')
            # Convert binary string to integer
            return int(binary_str, 2)
        else:
            raise TypeError(f"Code must be an integer or binary string, got {type(code_input)}")
    
    @property
    @abstractmethod
    def min_value(self) -> float:
        """Minimum representable value."""
        pass
    
    @property
    @abstractmethod
    def max_value(self) -> float:
        """Maximum representable value."""
        pass
    
    @property
    def total_codes(self) -> int:
        """Total number of possible codes."""
        return 1 << self.total_bits
    
    @abstractmethod
    def get_value_at_code(self, code: Union[int, str]) -> NumericValue:
        """Get the numeric value for a specific code."""
        pass
    
    @abstractmethod
    def enumerate_values(self) -> Iterator[NumericValue]:
        """Enumerate all possible values in the format."""
        pass

    @abstractmethod
    def count_values_between(self, min_val: float, max_val: float) -> int:
        """Count values within a specific range, analytically"""
        pass
    
    def get_min_normal(self) -> Optional[float]:
        """Minimum normal value (for floating-point formats)."""
        return None
    
    def get_max_normal(self) -> Optional[float]:
        """Maximum normal value (for floating-point formats)."""
        return None
    
    def get_min_subnormal(self) -> Optional[float]:
        """Minimum subnormal value (for floating-point formats)."""
        return None
    
    def get_max_subnormal(self) -> Optional[float]:
        """Maximum subnormal value (for floating-point formats)."""
        return None
    
    def __str__(self) -> str:
        return f"{self.name} ({self.total_bits}-bit)"


class INTFormat(NumericFormat):
    """Integer numeric format."""
    
    def __init__(self, name: str, total_bits: int, signed: bool = True):
        super().__init__(name, total_bits)
        self.signed = signed
        self._min_value = self._calculate_min_value()
        self._max_value = self._calculate_max_value()
    
    def _calculate_min_value(self) -> float:
        """Calculate minimum representable value."""
        if self.signed:
            return -(1 << (self.total_bits - 1))
        else:
            return 0
    
    def _calculate_max_value(self) -> float:
        """Calculate maximum representable value."""
        if self.signed:
            return (1 << (self.total_bits - 1)) - 1
        else:
            return (1 << self.total_bits) - 1
    
    @property
    def min_value(self) -> float:
        """Minimum representable value."""
        return self._min_value
    
    @property
    def max_value(self) -> float:
        """Maximum representable value."""
        return self._max_value
    
    def get_value_at_code(self, code: Union[int, str]) -> NumericValue:
        """Get the integer value for a specific code."""
        # Normalize the input code
        code_int = self._normalize_code_input(code)
        
        if not (0 <= code_int < self.total_codes):
            raise ValueError(f"Code {code} out of range for {self.total_bits}-bit format")
        
        # Convert to signed if necessary
        if self.signed:
            # Two's complement conversion
            if code_int & (1 << (self.total_bits - 1)):
                # Negative number
                value = code_int - (1 << self.total_bits)
            else:
                # Positive number
                value = code_int
        else:
            value = code_int
        
        category = "integer"
        if value == 0:
            category = "zero"
        
        return NumericValue(
            code=code_int,
            value=float(value),
            code_binary=f"{code_int:0{self.total_bits}b}",
            category=category
        )
    
    def enumerate_values(self) -> Iterator[NumericValue]:
        """Enumerate all possible integer values."""
        if self.total_bits > 16:
            raise ValueError(f"Total bits {self.total_bits} is too large for enumeration")
        for code in range(self.total_codes):
            yield self.get_value_at_code(code)
    

    def count_values_between(self, min_val: float, max_val: float) -> int:
        """Count integer values within a specific range, analytically."""
        # Clamp min_val to the representable range
        min_int = max(self.min_value, math.ceil(min_val))
        # Clamp max_val to the representable range  
        max_int = min(self.max_value, math.floor(max_val))
        
        # Count the integers in the range [min_int, max_int]
        if min_int > max_int:
            return 0
        return max_int - min_int + 1





class FPFormat(NumericFormat):
    """Floating-point numeric format.
    **IEEE754** floating-point rules:
    - 1 sign bit, e exponent bits, m mantissa bits
    - Exponent bias default is B = 2^(e-1) - 1
    - Normal numbers: stored exponent != 0 and != s.11..1 (all 1s), value = (-1)^s * 2^(E - B) * (1 + frac/2^m)
    - Subnormals: stored exponent == 0, value = (-1)^s * 2^(1 - B) * (0 + frac/2^m)
    - Zeros: stored exponent == 0 and frac == 0 -> ±0 (unless reserved)
    - Infinities: stored exponent == s.11..1 and frac == 0 -> ±∞.
    - NaN: stored exponent == s.11..1 and frac != 0 -> NaN.
    """
    
    def __init__(self, name: str, exponent_bits: int, mantissa_bits: int, 
                 bias: Optional[int] = None):
        total_bits = 1 + exponent_bits + mantissa_bits  # sign + exp + mantissa
        super().__init__(name, total_bits)
        
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.bias = bias if bias is not None else ((1 << (exponent_bits - 1)) - 1)
        
        # Pre-calculate analytical properties
        self._max_exp = (1 << exponent_bits) - 1
        self._calculate_ranges()
    
    def _calculate_ranges(self):
        """Calculate all the analytical ranges for the floating-point format."""
        # Normal numbers
        self._min_normal = 2.0 ** (1 - self.bias)
        self._max_normal = 2.0 ** (self._max_exp - 1 - self.bias) * (2.0 - 1.0 / (1 << self.mantissa_bits))
        
        # Subnormal numbers
        self._min_subnormal = 2.0 ** (1 - self.bias) / (1 << self.mantissa_bits)
        self._max_subnormal = 2.0 ** (1 - self.bias) * (1.0 - 1.0 / (1 << self.mantissa_bits))
        
        # Overall min/max
        self._min_value = -self._max_normal  # Most negative normal
        self._max_value = self._max_normal   # Most positive normal
    
    @property
    def min_value(self) -> float:
        """Minimum representable finite value."""
        return self._min_value
    
    @property
    def max_value(self) -> float:
        """Maximum representable finite value."""
        return self._max_value
    
    def get_min_normal(self) -> float:
        """Minimum normal value."""
        return self._min_normal
    
    def get_max_normal(self) -> float:
        """Maximum normal value."""
        return self._max_normal
    
    def get_min_subnormal(self) -> float:
        """Minimum subnormal value."""
        return self._min_subnormal
    
    def get_max_subnormal(self) -> float:
        """Maximum subnormal value."""
        return self._max_subnormal
    
    @property
    def machine_epsilon(self) -> float:
        """Machine epsilon for normal numbers."""
        return 2.0 ** (-self.mantissa_bits)
    
    @property
    def ulp_at_1(self) -> float:
        """Unit in the last place at 1.0."""
        return 2.0 ** (-self.mantissa_bits)
    
    def get_value_at_code(self, code: Union[int, str]) -> NumericValue:
        """Get the floating-point value for a specific code."""
        # Normalize the input code
        code_int = self._normalize_code_input(code)
        
        if not (0 <= code_int < self.total_codes):
            raise ValueError(f"Code {code} out of range for {self.total_bits}-bit format")
        
        # Extract bit fields
        self.sign = (code_int >> (self.exponent_bits + self.mantissa_bits)) & 0x1
        self.exp_stored = (code_int >> self.mantissa_bits) & ((1 << self.exponent_bits) - 1)
        self.frac = code_int & ((1 << self.mantissa_bits) - 1)
        
        return self._intcode2float(code_int)


    def _intcode2float(self, code: int) -> float:
        """
        convert integer code to floating-point value, following IEEE754 rules.
        """
        if self.exp_stored == 0:
            # Zero or subnormal
            if self.frac == 0:
                value = -0.0 if self.sign else 0.0
                category = "zero"
            else:
                exponent = 1 - self.bias
                mantissa = self.frac / (1 << self.mantissa_bits)
                value = ((-1.0) ** self.sign) * (2.0 ** exponent) * mantissa
                category = "subnormal"
        elif self.exp_stored == self._max_exp:
            # Infinity or NaN
            if self.frac == 0:
                value = float('-inf') if self.sign else float('inf')
                category = "infinity"
            else:
                value = float('nan')
                category = "nan"
        else:
            # Normal number
            exponent = self.exp_stored - self.bias
            mantissa = 1.0 + self.frac / (1 << self.mantissa_bits)
            value = ((-1.0) ** self.sign) * (2.0 ** exponent) * mantissa
            category = "normal"
        
        return NumericValue(
            code=code,
            value=value,
            code_binary=f"{code:0{self.total_bits}b}",
            category=category
        )

    def _float2intcode(self, value: float, direction: str = "inner") -> int:
        """Convert a float value to its IEEE754 binary representation.
        Args:
            value (float): The floating-point value to convert
            direction (str): "inner" or "outer" towards zero,
        Returns:
            int: The binary code representation as an integer
        """
        # Handle special cases first
        if math.isnan(value):
            # For NaN, set all exponent bits to 1 and non-zero fraction
            return (self._max_exp << self.mantissa_bits) | 1
            
        if math.isinf(value):
            # For infinity, set all exponent bits to 1 and fraction to 0
            sign_bit = 1 if value < 0 else 0
            return (sign_bit << (self.exponent_bits + self.mantissa_bits)) | \
                   (self._max_exp << self.mantissa_bits)
            
        if value == 0.0:
            # Handle both +0 and -0
            sign_bit = 1 if math.copysign(1, value) < 0 else 0
            return sign_bit << (self.exponent_bits + self.mantissa_bits)
            
        # Handle normal process for non-zero finite numbers
        # 1. Determine sign bit
        sign_bit = 1 if value < 0 else 0
        abs_value = abs(value)
        # 2. Calculate exponent and check for subnormal numbers
        exp = math.floor(math.log2(abs_value))  # Always use floor for exponent      
        if exp < (1 - self.bias):  # Subnormal number
            # For subnormal numbers:
            # - stored exponent is 0
            # - mantissa is shifted to fit
            stored_exp = 0
            mantissa = abs_value / (2.0 ** (1 - self.bias))
        else:  # Normal number
            stored_exp = exp + self.bias
            if stored_exp >= self._max_exp:  # Would overflow to infinity
                return (sign_bit << (self.exponent_bits + self.mantissa_bits)) | \
                       (self._max_exp << self.mantissa_bits)
            mantissa = abs_value / (2.0 ** exp) - 1.0
        # 3. Calculate fraction bits with directional rounding
        if direction == "inner":
            # Round towards zero
            frac = math.floor(mantissa * (1 << self.mantissa_bits))
        else:  # "outer"
            # Round away from zero
            frac = math.ceil(mantissa * (1 << self.mantissa_bits))
        
        # 4. Combine all parts into final binary representation
        code = (sign_bit << (self.exponent_bits + self.mantissa_bits)) | \
               (stored_exp << self.mantissa_bits) | \
               frac
        return code

    
    def enumerate_values(self) -> Iterator[NumericValue]:
        """Enumerate all possible floating-point values."""
        if self.total_bits > 16:
            raise ValueError(f"Total bits {self.total_bits} is too large for enumeration")
        for code in range(self.total_codes):
            yield self.get_value_at_code(code)
    

    def count_values_between(self, min_val: float, max_val: float) -> int:
        """Count values within a specific range [min_val, max_val], inclusive.
        
        Args:
            min_val (float): Lower bound (inclusive)
            max_val (float): Upper bound (inclusive)
            
        Returns:
            int: Number of representable values in the range
            
        Raises:
            ValueError: If min_val > max_val
        """
        if min_val > max_val:
            raise ValueError("min_val must be less than or equal to max_val")
        # Handle special cases
        if math.isnan(min_val) or math.isnan(max_val):
            return 0
        # If range includes infinity, adjust to max normal value
        if math.isinf(max_val):
            max_val = self._max_normal
        if math.isinf(min_val):
            min_val = -self._max_normal
        # Clamp to format's representable range
        min_val = max(min_val, -self._max_normal)
        max_val = min(max_val, self._max_normal)
        if min_val < 0 and max_val > 0:
            return self.count_values_between(min_val, -0.0) + self.count_values_between(+0.0, max_val)
        elif min_val < 0 and max_val == 0:
            return self.count_values_between(min_val, -self._min_subnormal) + 1 # +1 for -0.0
        elif min_val == 0 and max_val > 0:
            return self.count_values_between(self._min_subnormal, max_val) + 1 # +1 for +0.0
        elif min_val < 0 and max_val <0: # min_val and max_val on the same side of 0, so the int format shall have same sign
            code_min = self._float2intcode(min_val, direction="inner")
            code_max = self._float2intcode(max_val, direction="outer")
            return abs(code_max - code_min) + 1
        elif min_val > 0 and max_val > 0:
            code_min = self._float2intcode(min_val, direction="outer")
            code_max = self._float2intcode(max_val, direction="inner")
            return abs(code_max - code_min) + 1
        else:
            return 0




class FPFNFormat(FPFormat):
    """FPFN floating-point numeric format.
    NOTE: General FN (finite variant) rules. FP8_E4M3FN and FP8_E5M2FN are current use cases. Other bit width (other than 8-bit) are possible.
    ONNX FP8 rules: 
    Ref: [ONNX Guide](https://onnx.ai/onnx/technical/float8.html)
    Ref: [AdamSawicki Article](https://asawicki.info/articles/fp8_tables.php)
    Ref: [OCP FP8 Specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1)
    FP8 Formats for Deep Learning from NVIDIA, Intel, ARM and OCP introduces two types following IEEE specification.
    - 1 sign bit, e exponent bits, m mantissa bits.
    - Exponent bias default is B = 2^(e-1) - 1
    - Normal numbers: stored exponent != 0 and != s.11..1 (all 1s), value = (-1)^s * 2^(E - B) * (1 + frac/2^m)
    - Subnormals: stored exponent == 0, value = (-1)^s * 2^(1 - B) * (0 + frac/2^m)
    - Zeros: stored exponent == 0 and frac == 0, s.00..0.00..0 -> ±0
    - Infinities: No inf (Where the FN finite abbreviation comes from)
    - NaN: stored exponent == s.11..1 and frac = 1..1 ->  ±NaN
    """
    
    
    def __init__(self, name: str, exponent_bits: int, mantissa_bits: int, bias: Optional[int] = None):
        super().__init__(name, exponent_bits, mantissa_bits, bias)


    def _intcode2float(self, code: int) -> float:
        """
        convert integer code to floating-point value, using FN rules.
        """
        if self.exp_stored == 0:
            if self.frac == 0:
                value = -0.0 if self.sign else 0.0
                category = "zero"
            else:
                exponent = 1 - self.bias
                mantissa = self.frac / (1 << self.mantissa_bits)
                value = ((-1.0) ** self.sign) * (2.0 ** exponent) * mantissa
                category = "subnormal"
        elif self.exp_stored == self._max_exp and self.frac == (1 << self.mantissa_bits) - 1:
            value = float('nan')
            category = "nan"
        else:
            exponent = self.exp_stored - self.bias
            mantissa = 1.0 + self.frac / (1 << self.mantissa_bits)
            value = ((-1.0) ** self.sign) * (2.0 ** exponent) * mantissa
            category = "normal"
        
        return NumericValue(
            code=code,
            value=value,
            code_binary=f"{code:0{self.total_bits}b}",
            category=category
        )


    def _float2intcode(self, value: float, direction: str = "inner") -> int:
        """Convert a float value to its binary representation, using FN rules.
        direction: "inner" or "outer" towards zero,
        """
        # Handle special cases first
        if math.isnan(value):
            # For NaN, set all exponent bits and mantissa bits to 1
            return (1 << (self.exponent_bits + self.mantissa_bits)) - 1
        if math.isinf(value):
            # No infinity in FN format
            raise ValueError("Infinity is not representable in FN format")
        if value == 0.0:
            # Handle both +0 and -0
            sign_bit = 1 if math.copysign(1, value) < 0 else 0
            return sign_bit << (self.exponent_bits + self.mantissa_bits)
        # Handle normal process for non-zero finite numbers
        # 1. Determine sign bit
        sign_bit = 1 if value < 0 else 0
        abs_value = abs(value)
        # 2. Calculate exponent and check for subnormal numbers
        exp = math.floor(math.log2(abs_value))
        if exp < (1 - self.bias):  # Subnormal number
            # For subnormal numbers:
            # - stored exponent is 0
            # - mantissa is shifted to fit
            stored_exp = 0
            mantissa = abs_value / (2.0 ** (1 - self.bias))
        else:  # Normal number
            stored_exp = exp + self.bias
            if stored_exp >= self._max_exp:  # Would overflow to infinity
                return (sign_bit << (self.exponent_bits + self.mantissa_bits)) | \
                       (self._max_exp << self.mantissa_bits)
            mantissa = abs_value / (2.0 ** exp) - 1.0
        # 3. Calculate fraction bits
        if direction == "inner":
            frac = math.floor(mantissa * (1 << self.mantissa_bits))
        else:
            frac = math.ceil(mantissa * (1 << self.mantissa_bits))
        # 4. Combine all parts into final binary representation
        code = (sign_bit << (self.exponent_bits + self.mantissa_bits)) | \
               (stored_exp << self.mantissa_bits) | \
               frac
        return code




    def _calculate_ranges(self):
        """Calculate all the analytical ranges for the FN format."""
        self._min_normal = 2.0 ** (1 - self.bias)
        self._max_normal = 2.0 ** (self._max_exp - self.bias) * (2.0 - 1.0 / (1 << (self.mantissa_bits - 1))) # note that exponent can be all 1s, and mantissa_bits -1 to exclude the mantissa all 1 case, which is defined as NaN here
        
        self._min_subnormal = 2.0 ** (1 - self.bias) / (1 << self.mantissa_bits)
        self._max_subnormal = 2.0 ** (1 - self.bias) * (1.0 - 1.0 / (1 << self.mantissa_bits))
        
        self._min_value = -self._max_normal  # Most negative normal
        self._max_value = self._max_normal   # Most positive normal

        



class FPFNUZFormat(FPFormat):
    """FPFNUZ floating-point numeric format.
    NOTE: General FNUZ (finite + unsigned variant) rules. FP8_E4M3FNUZ and FP8_E5M2FNUZ are current use cases. Other bit width (other than 8-bit) are possible.
    ONNX FP8 rules: 
    Ref: [ONNX Guide](https://onnx.ai/onnx/technical/float8.html)
    Ref: [AdamSawicki Article](https://asawicki.info/articles/fp8_tables.php)
    Paper "8-bit Numerical Formats For Deep Neural Networks" by Graphcore introduces similar types.
    - 1 sign bit, e exponent bits, m mantissa bits.
    - Exponent bias default is B = 2^(e-1) [DIFFERENT FROM DEFAULT BIASES OF FP AND FN FORMATS]
    - Normal numbers: stored exponent != 0 and != s.11..1 (all 1s), value = (-1)^s * 2^(E - B) * (1 + frac/2^m)
    - Subnormals: stored exponent == 0, value = (-1)^s * 2^(1 - B) * (0 + frac/2^m)
    - Zeros: signed ==0, stored exponent == 0 and frac == 0, 0.00..0.00..0 -> 0
    - Infinities: No inf (Where the FN finite abbreviation comes from)
    - NaN: change the sign bit to 1 from zero, 1.00..0.00..0 -> NaN
    """
    
    def __init__(self, name: str, exponent_bits: int, mantissa_bits: int, bias: Optional[int] = None):
        self.bias = bias if bias is not None else (1 << (exponent_bits - 1))
        super().__init__(name, exponent_bits, mantissa_bits, self.bias)

    def _intcode2float(self, code: int) -> float:
        """
        convert integer code to floating-point value, using FNUZ rules.
        """
        if self.exp_stored == 0:
            if self.frac == 0:
                if self.sign == 0:
                    value = 0.0
                    category = "zero"
                else:
                    value = float('nan')
                    category = "nan"
            else:
                exponent = 1 - self.bias
                mantissa = self.frac / (1 << self.mantissa_bits)
                value = ((-1.0) ** self.sign) * (2.0 ** exponent) * mantissa
                category = "subnormal"
        else:
            exponent = self.exp_stored - self.bias
            mantissa = 1.0 + self.frac / (1 << self.mantissa_bits)
            value = ((-1.0) ** self.sign) * (2.0 ** exponent) * mantissa
            category = "normal"
        
        return NumericValue(
            code=code,
            value=value,
            code_binary=f"{code:0{self.total_bits}b}",
            category=category
        )


    def _float2intcode(self, value: float, direction: str = "inner") -> int:
        """Convert a float value to its binary representation, using FNUZ rules.
        direction: "inner" or "outer" towards zero,
        """
        # Handle special cases first
        if math.isnan(value):
            # For NaN, the format is 1.00..0.00..0
            return (1 << (self.exponent_bits + self.mantissa_bits))
        if math.isinf(value):
            # No infinity in FNUZ format
            raise ValueError("Infinity is not representable in FNUZ format")
        if value == 0.0:
            # For 0.0, the format is 0.00..0.00..0
            return 0 << (self.exponent_bits + self.mantissa_bits)
        # Handle normal process for non-zero finite numbers
        # 1. Determine sign bit
        sign_bit = 1 if value < 0 else 0
        abs_value = abs(value)
        # 2. Calculate exponent and check for subnormal numbers
        exp = math.floor(math.log2(abs_value))
        if exp < (1 - self.bias):  # Subnormal number
            # For subnormal numbers:
            # - stored exponent is 0
            # - mantissa is shifted to fit
            stored_exp = 0
            mantissa = abs_value / (2.0 ** (1 - self.bias))
        else:  # Normal number
            stored_exp = exp + self.bias
            if stored_exp >= self._max_exp:  # Would overflow to infinity
                return (sign_bit << (self.exponent_bits + self.mantissa_bits)) | \
                       (self._max_exp << self.mantissa_bits)
            mantissa = abs_value / (2.0 ** exp) - 1.0
        # 3. Calculate fraction bits
        if direction == "inner":
            frac = math.floor(mantissa * (1 << self.mantissa_bits))
        else:
            frac = math.ceil(mantissa * (1 << self.mantissa_bits))
        # 4. Combine all parts into final binary representation
        code = (sign_bit << (self.exponent_bits + self.mantissa_bits)) | \
               (stored_exp << self.mantissa_bits) | \
               frac
        return code

    def _calculate_ranges(self):
        """Calculate all the analytical ranges for the FNUZ format."""
        self._min_normal = 2.0 ** (1 - self.bias)
        self._max_normal = 2.0 ** (self._max_exp - self.bias) * (2.0 - 1.0 / (1 << self.mantissa_bits))
        
        self._min_subnormal = 2.0 ** (1 - self.bias) / (1 << self.mantissa_bits)
        self._max_subnormal = 2.0 ** (1 - self.bias) * (1.0 - 1.0 / (1 << self.mantissa_bits))
        
        self._min_value = -self._max_normal  # Most negative normal
        self._max_value = self._max_normal   # Most positive normal



class HIF8Format(NumericFormat):
    """HIF8 floating-point numeric format.
    
    HiFloat8 is a non-standard 8-bit floating-point format with:
    - 1 sign bit
    - Variable-length prefix codes for exponent field (Dot field)
    - Variable mantissa widths based on Dot field value
    - Special values: Zero (0x00), NaN (0x80), Infinity (±0x6F, ±0xEF)
    - Denormal numbers with special encoding
    - Sign-magnitude exponent encoding with implicit leading '1'
    """
    
    def __init__(self, name: str):
        super().__init__(name, 8)
        self._calculate_ranges()
    
    def _calculate_ranges(self):
        """Calculate min/max values by examining all possible codes."""
        min_val = float('inf')
        max_val = float('-inf')
        
        for code in range(256):
            try:
                numeric_val = self.get_value_at_code(code)
                if not math.isnan(numeric_val.value) and not math.isinf(numeric_val.value):
                    min_val = min(min_val, abs(numeric_val.value))
                    max_val = max(max_val, abs(numeric_val.value))
            except:
                continue
        
        self._min_value = -max_val
        self._max_value = max_val
    
    @property
    def min_value(self) -> float:
        """Minimum representable finite value."""
        return self._min_value
    
    @property
    def max_value(self) -> float:
        """Maximum representable finite value."""
        return self._max_value
    
    def get_value_at_code(self, code: Union[int, str]) -> NumericValue:
        """Get the HiFloat8 value for a specific code."""
        # Normalize the input code
        code_int = self._normalize_code_input(code)
        
        if not (0 <= code_int < 256):
            raise ValueError(f"Code {code} out of range for 8-bit format")
        
        # Use the decode_hif8 logic
        bits = f'{code_int:08b}'
        sign_bit = bits[0]
        sign_val = -1.0 if sign_bit == '1' else 1.0
        
        # Handle special values
        if code_int == 0b00000000:
            return NumericValue(
                code=code_int,
                value=0.0,
                code_binary=bits,
                category="zero"
            )
        
        if code_int == 0b10000000:
            return NumericValue(
                code=code_int,
                value=float('nan'),
                code_binary=bits,
                category="nan"
            )
        
        if code_int == 0b01101111:
            return NumericValue(
                code=code_int,
                value=float('inf'),
                code_binary=bits,
                category="infinity"
            )
        
        if code_int == 0b11101111:
            return NumericValue(
                code=code_int,
                value=float('-inf'),
                code_binary=bits,
                category="infinity"
            )
        
        # Parse Dot field using prefix codes
        dot_code = None
        em_bits = ''
        m_bits = ''
        
        # Case D=4 (Dot='11')
        if bits[1:3] == '11':
            dot_code, D, m_width = '11', 4, 1
            em_bits, m_bits = bits[3:7], bits[7:]
        # Case D=3 (Dot='10')
        elif bits[1:3] == '10':
            dot_code, D, m_width = '10', 3, 2
            em_bits, m_bits = bits[3:6], bits[6:]
        # Case D=2 (Dot='01')
        elif bits[1:3] == '01':
            dot_code, D, m_width = '01', 2, 3
            em_bits, m_bits = bits[3:5], bits[5:]
        # Case D=1 (Dot='001')
        elif bits[1:4] == '001':
            dot_code, D, m_width = '001', 1, 3
            em_bits, m_bits = bits[4:5], bits[5:]
        # Case D=0 (Dot='0001')
        elif bits[1:5] == '0001':
            dot_code, D, m_width = '0001', 0, 3
            m_bits = bits[5:]
        # Case Denormal (Dot='0000')
        elif bits[1:5] == '0000':
            m_bits = bits[5:]
            m_val = int(m_bits, 2)
            # Denormal formula: X = (-1)^S * 2^(M-23) * 1.0
            exponent = m_val - 23
            value = sign_val * (2.0**exponent)
            return NumericValue(
                code=code_int,
                value=value,
                code_binary=bits,
                category="denormal"
            )
        else:
            # Invalid encoding
            return NumericValue(
                code=code_int,
                value=float('nan'),
                code_binary=bits,
                category="invalid"
            )
        
        # Calculate normal value
        # Decode exponent from sign-magnitude Em with implicit leading '1'
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
        
        return NumericValue(
            code=code_int,
            value=value,
            code_binary=bits,
            category="normal"
        )
    
    def enumerate_values(self) -> Iterator[NumericValue]:
        """Enumerate all possible HiFloat8 values."""
        for code in range(256):
            yield self.get_value_at_code(code)
    
    def count_values_between(self, min_val: float, max_val: float) -> int:
        """Count values within a specific range [min_val, max_val], inclusive."""
        if min_val > max_val:
            raise ValueError("min_val must be less than or equal to max_val")
        
        count = 0
        for code in range(256):
            try:
                numeric_val = self.get_value_at_code(code)
                if (not math.isnan(numeric_val.value) and 
                    min_val <= numeric_val.value <= max_val):
                    count += 1
            except:
                continue
        
        return count





# Example usage and testing
if __name__ == "__main__":
   
    # Floating-point formats
    fp_formats = {
        'FP4_E2M1': FPFormat('FP4_E2M1', 2, 1),  # E2M1 format
        'FP8_E4M3': FPFormat('FP8_E4M3', 4, 3),  # Standard E4M3
        'FP8_E5M2': FPFormat('FP8_E5M2', 5, 2),  # E5M2 format
        'FP16': FPFormat('FP16', 5, 10),         # Half precision
        'BF16': FPFormat('BF16', 8, 7),          # BFloat16
        'FP32': FPFormat('FP32', 8, 23),         # Single precision
        'FP64': FPFormat('FP64', 11, 52),        # Double precision
        'FP8_E4M3FN': FPFNFormat('FP8_E4M3FN', 4, 3),             # FP8_E4M3FN
        'FP8_E4M3FNUZ': FPFNUZFormat('FP8_E4M3FNUZ', 4, 3),         # FP8_E4M3FNUZ
    }
    


    import matplotlib.pyplot as plt
    import numpy as np
    fp16 = fp_formats['FP16']
    vals = fp16.enumerate_values()
    # output values to csv
    import pandas as pd
    df = pd.DataFrame([{
        'values': val.value,
        'categories': val.category,
        'code_binary': val.code_binary,
        'code': val.code
    } for val in vals])
    df.to_csv('fp16_values.csv', index=False)
    