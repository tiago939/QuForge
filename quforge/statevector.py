import numpy as np
import sympy as sp
import torch
from IPython.display import display, Math, Latex

class LatexDisplay:
    def __init__(self, latex):
        self.latex = latex

    def _repr_latex_(self):
        return f"$$ {self.latex} $$"
    
    def __str__(self):
        return self.latex

def show(state, dim=2, wires=1, tol=1e-12, use_floats=False, 
                        float_precision=4, emphasize_index=None, suppress_inner=False):
    """
    Convert a state vector into a LaTeX formatted string.
    
    Parameters:
        state (np.ndarray): The state vector as a 1D numpy array. (If using a tensor library,
                            you may need to detach and convert it as shown.)
        dim (int): The dimension of each qudit (default is 2 for qubits).
        wires (int): The number of qudits in the circuit.
        tol (float): Tolerance below which amplitudes are considered zero.
        use_floats (bool): If True, display amplitudes as floating point numbers.
        float_precision (int): The number of significant digits for floats.
        emphasize_index (int or None): If an integer is provided, the specified index
                                       (0-indexed) is factored out and emphasized.
                                       If None, no emphasis is applied.
        suppress_inner (bool): If True (and emphasize_index is not None), the non-emphasized
                               part is replaced by a placeholder ket (e.g. |\phi_1\rangle,
                               |\phi_2\rangle, etc.) instead of showing its full expansion.
                               Default is False.
    
    Returns:
        str: A string containing the LaTeX representation of the state.
    """
    # Convert state to a 1D numpy array.
    try:
        state = state.detach().cpu().numpy().flatten()
    except AttributeError:
        state = np.array(state).flatten()
    
    # If no emphasis is requested, do the normal conversion.
    if emphasize_index is None:
        terms = []
        for idx, amp in enumerate(state):
            if np.abs(amp) < tol:
                continue

            # Format amplitude.
            if use_floats:
                re = amp.real
                im = amp.imag
                if np.abs(im) < tol:
                    amp_str = f"{re:.{float_precision}g}"
                elif np.abs(re) < tol:
                    amp_str = f"{im:.{float_precision}g}i"
                else:
                    sign = '+' if im >= 0 else '-'
                    amp_str = f"{re:.{float_precision}g}{sign}{abs(im):.{float_precision}g}i"
            else:
                if np.abs(amp.imag) < tol:
                    amp = amp.real
                simplified_amp = sp.nsimplify(amp, [sp.sqrt(2)], tolerance=1e-10)
                if isinstance(simplified_amp, sp.Rational):
                    simplified_amp = simplified_amp.limit_denominator(1000)
                amp_str = sp.latex(simplified_amp)
                if amp_str == '1':
                    amp_str = ''
                elif amp_str == '-1':
                    amp_str = '-'
            
            # Convert the index into a ket string.
            ket_digits = []
            for i in range(wires):
                digit = (idx // (dim ** (wires - i - 1))) % dim
                ket_digits.append(str(digit))
            ket_str = "".join(ket_digits)
            
            terms.append(f"{amp_str}|{ket_str}\\rangle")
            
        latex_state = " + ".join(terms)
        latex_state = latex_state.replace("+ -", "- ")
        return latex_state

    # If emphasis is requested, factor out the specified qudit.
    else:
        groups = {}  # Group terms by the emphasized digit.
        for idx, amp in enumerate(state):
            if np.abs(amp) < tol:
                continue

            # Format amplitude.
            if use_floats:
                re = amp.real
                im = amp.imag
                if np.abs(im) < tol:
                    amp_str = f"{re:.{float_precision}g}"
                elif np.abs(re) < tol:
                    amp_str = f"{im:.{float_precision}g}i"
                else:
                    sign = '+' if im >= 0 else '-'
                    amp_str = f"{re:.{float_precision}g}{sign}{abs(im):.{float_precision}g}i"
            else:
                if np.abs(amp.imag) < tol:
                    amp = amp.real
                simplified_amp = sp.nsimplify(amp, [sp.sqrt(2)], tolerance=1e-10)
                if isinstance(simplified_amp, sp.Rational):
                    simplified_amp = simplified_amp.limit_denominator(1000)
                amp_str = sp.latex(simplified_amp)
                if amp_str == '1':
                    amp_str = ''
                elif amp_str == '-1':
                    amp_str = '-'

            # Build the full ket as a list of digits.
            digits = []
            for i in range(wires):
                digit = (idx // (dim ** (wires - i - 1))) % dim
                digits.append(str(digit))
            
            # Extract the digit to emphasize.
            emph_digit = digits[emphasize_index]
            # Build the remaining ket (removing the emphasized digit).
            remaining_digits = digits[:emphasize_index] + digits[emphasize_index+1:]
            remaining_ket = "".join(remaining_digits)
            
            # Group terms by the emphasized digit.
            if emph_digit not in groups:
                groups[emph_digit] = []
            groups[emph_digit].append((amp_str, remaining_ket))
        
        # Build the grouped LaTeX string.
        group_terms = []
        placeholder_counter = 0
        for digit, term_list in groups.items():
            if suppress_inner:
                # Replace the detailed inner sum with a placeholder ket.
                inner = f"|\\phi_{{{placeholder_counter}}}\\rangle"
                placeholder_counter += 1
            else:
                inner_terms = []
                for (amp_str, rem_ket) in term_list:
                    # If there are no remaining digits, just show the amplitude.
                    if rem_ket == "":
                        inner_terms.append(f"{amp_str}")
                    else:
                        inner_terms.append(f"{amp_str}|{rem_ket}\\rangle")
                inner = "(" + " + ".join(inner_terms).replace("+ -", "- ") + ")"
            group_terms.append(f"|{digit}\\rangle\\,{inner}")
        
        latex_state = " + ".join(group_terms)
        latex_state = latex_state.replace("+ -", "- ")

        return latex_state