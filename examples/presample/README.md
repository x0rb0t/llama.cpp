# llama.cpp/examples/presample

## Overview

This directory contains the `presample` example implementation for using `llama.cpp`. The `presample` example demonstrates how to integrate advanced reflective reasoning and uncertainty handling mechanisms into an AI chatbot using the `llama.cpp` library. The chatbot employs structured thinking processes to address complex queries and uncertain contexts, making it ideal for analytical tasks.

## Key Features

- **Reflective Reasoning**: The chatbot alternates between active dialogue and deeper reasoning phases, using predefined tags for structured analysis.
- **Uncertainty Handling**: Dynamically transitions into reasoning mode when uncertainty thresholds are exceeded.
- **Customizable Configurations**: Includes a range of parameters like temperature, token limits, and continuation attempts to fine-tune the chatbot’s behavior.
- **Verbose Output Modes**: Offers different levels of output verbosity (verbose, dots, silent) for debugging or user preferences.

## File Structure

- **`llama_chain.hpp`**: A header file containing utilities to manage token chains and facilitate reasoning.
- **`llama.h`**: Core llama library for model loading and token processing.
- **`presample.cpp`**: Main implementation file demonstrating advanced chatbot functionality.

## Prerequisites

To run this example, ensure you have:

1. A compiled version of `llama.cpp`.
2. A llama-compatible model file.
3. A C++17 or later compiler.

## Usage

Run the example program with the following syntax:

```bash
./presample --model <path-to-model> [options]
```

### Command-line Options

| Option                        | Description                                 | Default Value |
|-------------------------------|---------------------------------------------|---------------|
| `--model <path>`              | Path to the model file (required).         | -             |
| `--gpu-layers <n>`            | Number of GPU layers to use.               | 99            |
| `--ctx-size <n>`              | Context size in tokens.                    | 8192          |
| `--batch-size <n>`            | Batch size for processing.                 | 8192          |
| `--uncertainty-threshold <n>` | Threshold for transitioning to reasoning.  | 2.5           |
| `--window-size <n>`           | Uncertainty window size.                   | 6             |
| `--thinking-temp <n>`         | Temperature for reasoning mode.            | 0.7           |
| `--continuation-temp <n>`     | Temperature for continuation generation.   | 0.75          |
| `--thinking-tokens <n>`       | Max tokens allowed for reasoning.          | 512           |
| `--continuation-tokens <n>`   | Tokens for each continuation attempt.      | 128           |
| `--thinking-attempts <n>`     | Number of reasoning attempts.              | 4             |
| `--continuation-attempts <n>` | Number of continuation attempts.           | 4             |
| `--max-tokens <n>`            | Max response tokens.                       | 1024          |
| `--temperature <n>`           | Sampling temperature.                      | 0.8           |
| `--top-k <n>`                 | Top-k sampling parameter.                  | 40            |
| `--thinking-output <mode>`    | Thinking output mode (`verbose`, `dots`, `silent`). | `verbose`    |

### Example Command

```bash
./presample --model ../models/7B/llama.bin --ctx-size 4096 --thinking-temp 0.8 --max-tokens 2048
```

## Chat Example

When you run the program, you can interact with the chatbot. Example:

```
Chat initialized with following parameters:
Context size: 8192
Batch size: 8192
Uncertainty threshold: 2.50
Temperature: 0.80
Thinking temperature: 1.00
Continuation temperature: 0.30
Thinking tokens: 512
Continuation tokens: 256
Thinking output mode: verbose

Enter your messages (Ctrl+D to exit):

User: Which number is bigger 9.11 or 9.9?

Assistant: <REFLECTION>
Nexus pauses to deeply analyze the immediate context and task requirements. With 512 tokens for analysis and 128 tokens for each continuation attempt, Nexus will structure its reasoning efficiently. Examining both recent user input and previous response (if any), Nexus breaks down the situation using systematic reasoning within <REASONING> tags. This space can be used freely for drafting and restructuring ideas, planning the next 128 tokens of response. The structured analysis must end with </REASONING>:
</REFLECTION>
<REASONING>
Breaking down and drafting key points:
1. **Compare the Whole Numbers First**: Both numbers, 9.11 and 9.9, have the same whole number part, which is 9.
2. **Compare the Decimal Parts Next**: 
   - Decimal part of 9.11: The first decimal place is 1, and the second decimal place (hundredth) is 1.
   - Decimal part of 9.9: The first decimal place is 9, and there is no number in the hundredth place (which we can consider as 0).
3. **Determining Which Number is Bigger**:
   - Comparing the tenths place directly: 1 (from 9.11) < 9 (from 9.9).
   - Even if we consider the hundredth place, adding a zero (making it 9.90), it remains clear that 1 < 9.


</REASONING>

>(uncertainty: 0.86)
>
1. Immediate requirement is to compare two decimal numbers, 9.11 and 9.9 (formatted as 9.90 for easy comparison),
2. Decimal places affect value representation and comparison.
3. Clear logical conclusion about which number is larger, based on standard decimal comparison rules.
</REASONING>

>(uncertainty: 1.57)
>
1. Understand numerical comparison - the values given are 9.11 and 9.9.
2. Evaluate each digit from left to right for comparison:
   - The whole number part (tens and units) of both numbers: Both are 9, so equal.
   - Compare the tenths place: 1 in 9.11 vs. 9 in 9.9.
   - Since 9 > 1, the comparison ends here without needing to check further decimal places.
</REASONING>

>(uncertainty: 1.27)
>
1. Comparison task for numerical values: 9.11 and 9.9
2. Direct numerical comparison is required
3. Decimal place consideration is essential
</REASONING>

>(uncertainty: 1.64)
>
1. Directly compare the decimal numbers 9.11 and 9.9.
2. Focus on comparing each digit starting from the tenths place.
</REASONING>

>(uncertainty: 1.15)
>
1. Numerical comparison rules in base 10.
2. Identifying place values and comparing digits from left to right.
3. Concluding the numerical relationship based on step-by-step comparison.
</REASONING>

>(uncertainty: 1.69)
>
<REFLECTION>
Based on the drafted analysis, planning next 128 tokens:
1. **Compare the Whole Numbers First**: Both numbers, 9.11 and 9.9, have the same whole number part, which is 9.
2. **Compare the Decimal Parts Next**: 
   - Decimal part of 9.11: The first decimal place is 1, and the second decimal place (hundredth) is 1.
   - Decimal part of 9.9: The first decimal place is 9, and there is no number in the hundredth place (which we can consider as 0).
3. **Determining Which Number is Bigger**:
   - Comparing the tenths place directly: 1 (from 9.11) < 9 (from 9.9).
   - Even if we consider the hundredth place, adding a zero (making it 9.90), it remains clear that 1 < 9.
Nexus will continue, integrating these refined insights naturally:
</REFLECTION>
</REASONING>
The number 9.9 is bigger than 9.11. When comparing the decimal parts, the first digit after the decimal point in 9.9 is 9, whereas in 9.11 it is 1, making 9.9 the larger number.
>(uncertainty: 0.47)
>
</REASONING>
The number 9.9 is bigger than 9.11. When comparing the decimal parts, the tenths place in 9.9 is 9, whereas in 9.11 it is 1, making 9.9 the larger number.
>(uncertainty: 0.47)
>
</REASONING>
The number 9.9 is bigger than 9.11. When comparing the decimal parts, the tenths place of 9.9 is 9, which is greater than the tenths place of 9.11, which is 1. Therefore, 9.9 is the larger number.
>(uncertainty: 0.42)
>
The number 9.9 is bigger than 9.11. When comparing the decimal parts, the tenths place of 9.9 is 9, which is greater than the tenths place of 9.11, which is 1. Therefore, 9.9 is the larger number.
```
