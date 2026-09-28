"""跨规约语言的 TFVR 内核。

一个统一的时序 IR(``tlast``)+ 一套离散时间轨迹语义(``monitor``)+ 一套 Z3 有界
编码(``smt``),让同一个忠实性判据(E-spec+ / E-spec-)在 LTL / STL / MTL /
ptLTL(FRETish) / SVA 上原样成立。表层语法差异全部收敛在 ``parse_*`` 前端里。
"""
