# Computing‑Cost Analysis

Below each heading you will find the original method, a short rationale of the dominant work that the interpreter performs, and the resulting **Big‑O / Ω** bound expressed as a function of *n* (the number of qubits, so the state‑vector length is 2ⁿ).

## __init__

```python
    def __init__(self, num_qubits):
        """
        Initializes the register to the specified number of qbits
        :param num_qubits: number of qbits
        """
        arr = np.zeros((2 ** num_qubits, 1), dtype=np.complex64)
        arr[0] = [1 + 0j]
        self.state = arr
        self.num_qubits = num_qubits
        self.rng = np.random.default_rng()
```

**Complexity reasoning**  
The constructor allocates a state‑vector of length 2ⁿ (one complex64 per amplitude). Array creation therefore
requires Θ(2ⁿ) operations and Θ(2ⁿ) bytes of memory. All subsequent scalar assignments and the random‑number‑generator 
instantiation are constant‑time.

**Result:** **Θ(2ⁿ)** time, **Θ(2ⁿ)** memory.

---

## apply_gate

```python
    def apply_gate(self, gate, target):
        """
        Receives a gate, and apply it to the target qbit. If the gate spans more than one qbit, it is assumed
        the target qbit is the first qbit to apply the gate to. Per example, having a 4qbit register, and applying
        a CNOT (2 qbits gate) in target qbit 2 (3rd qbit), results in the CNOT being applied to qbits 2 and 3.
        """
        self.__check_apply_gate_inputs(gate, target)
        gate_size = int(np.log2(gate.shape[0]))
        full_gate = self.__add_right_identity_gates(gate, target)
        full_gate = self.__add_left_identity_gates(full_gate, self.num_qubits - target - 1 - (gate_size - 1))
        self.state = full_gate.dot(self.state)
        return self
```

**Complexity reasoning**  
Starting from a *k*-qubit gate, the method repeatedly performs Kronecker 
products until it has a 2ⁿ×2ⁿ matrix, and then performs a dot product with the statevector.  
* If the gate is the "same" size as the statevector, then there is no need to add gates (__add_xyz methods). If not,
then each kron product of a 2x2 matrix with a 2^m*2^m matrix costs Θ((2^m)²). The last kron product determinate the 
computational cost (is the biggest), resulting in Θ((2^(n-1))²) = Θ(4^(n-1)).  
* The final matrix–vector multiplication is always Θ(4ⁿ).

**Result:** **Θ(4ⁿ) ≡ O(4ⁿ) ≡ Ω(4ⁿ)** time, with Θ(4ⁿ) memory to hold `full_gate`.

---

## qbit_prob

```python
    def qbit_prob(self, target):
        if target > self.num_qubits - 1:
            raise ValueError("Target qbit is outside the register. Target qbit: " + str(target))
        is_one = True
        period = 2 ** target
        total_prob = 0
        for i in range(0, self.state.size):
            if i % period == 0:
                is_one = not is_one
            if is_one:
                total_prob += self.value_prob(i)
        total_prob = self.__correct_probability_value(total_prob)
        return total_prob
```

**Complexity reasoning**  
`self.state.size` is 2ⁿ, and the loop touches every amplitude once, executing O(1) work per item. 
The modulus and additions do not depend on n otherwise.

**Result:** **Θ(2ⁿ)** time, **Θ(1)** extra memory.

---

## collapse

```python
    def collapse(self, target, value, prob_one=None):
        self.__check_collapse_inputs(target)
        is_one = True
        period = 2 ** target
        for i in range(0, self.state.size):
            if i % period == 0:
                is_one = not is_one
            if (is_one and value == 0) or (not is_one and value == 1):
                self.state[i] = 0
        if prob_one is None:
            norm = np.linalg.norm(self.state)
        elif value == 1:
            norm = prob_one ** 0.5
        else:
            norm = (1 - prob_one) ** 0.5
        for i in range(self.state.size):
            self.state[i] /= norm
        return self
```

**Complexity reasoning**  
Two sequential passes over the full 2ⁿ‑entry state‑vector (one to zero amplitudes, one to renormalise) each cost Θ(2ⁿ). 
Norm computation is linear as well. Constants do not change the bound.

**Result:** **Θ(2ⁿ) (time)**, **Θ(1)** extra memory (in‑place edits).

---

## measure

```python
    def measure(self, target, rng=None):
        prob_one = self.qbit_prob(target)     # Θ(2ⁿ)
        random_number = self.rng.random() if rng is None else rng.random()
        value = 1 if random_number < prob_one else 0
        self.collapse(target, value, prob_one)  # Θ(2ⁿ)
        return self, value
```

**Complexity reasoning**  
`measure` delegates almost all work to `qbit_prob` and `collapse`, each Θ(2ⁿ). 
The constant‑time random draw and comparison are negligible. Because the two calls are sequential, 
the costs add, keeping the same order.

**Result:** **Θ(2ⁿ)** time, **Θ(1)** extra memory.

