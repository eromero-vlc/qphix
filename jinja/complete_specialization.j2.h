{# Copyright © 2017 Martin Ueding <dev@martin-ueding.de> #}

{% macro include_generated_kernel(isa, kernel, operator, fptype, vec, soa, compress12) %}

{% if compress12 == '' %}
{% set suffix = '' %}
{% else %}
{% set suffix = '_12' if compress12 == 'true' else '_18' %}
{% endif %}

{% set filename = '%s/%s_%s_%s_%s_v%d_s%d%s'|format(isa, kernel, operator, fptype, fptype, vec, soa, suffix) %}
{% include filename %}
{% endmacro %}

{#
The following ex command can be used in Vim to replace all the C++ preprocessor
includes with the Jinja2 include.

%s#\v\#include INCLUDE_FILE_VAR\(qphix/avx2/generated/(.+),FPTYPE,VEC,SOA,COMPRESS_SUFFIX\)#// clang-format off{{ include_generated_kernel(ISA, "\1", FPTYPE, VEC, SOA, COMPRESS12) }}// clang-format on#
#}

#pragma once

#include "immintrin.h"
#include "qphix/geometry.h"
#include "qphix/avx/avx_utils.h"

{% include 'jinja/%s_general.j2.h'|format(kernel_base) %}

{% for FPTYPE, VEC, soalens in defines %}
{% set kernel = kernel_pattern|format(fptype=FPTYPE) %}
{% for SOA in soalens %}
{% for COMPRESS12 in ['true', 'false'] %}
#define FPTYPE {{ FPTYPE }}
#define VEC {{ VEC }}
#define SOA {{ SOA }}
#define COMPRESS12 {{ COMPRESS12 }}
{% include 'jinja/%s_specialization.j2.h'|format(kernel_base) %}
#undef FPTYPE
#undef VEC
#undef SOA
#undef COMPRESS12
{% endfor %}
{% endfor %}
{% endfor %}
