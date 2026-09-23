#include "sha256.h"

#include <string.h>

#define ROTR32(value, bits) (((value) >> (bits)) | ((value) << (32U - (bits))))

static const uint32_t sha256Constants[64] = {
    0x428A2F98UL,0x71374491UL,0xB5C0FBCFUL,0xE9B5DBA5UL,
    0x3956C25BUL,0x59F111F1UL,0x923F82A4UL,0xAB1C5ED5UL,
    0xD807AA98UL,0x12835B01UL,0x243185BEUL,0x550C7DC3UL,
    0x72BE5D74UL,0x80DEB1FEUL,0x9BDC06A7UL,0xC19BF174UL,
    0xE49B69C1UL,0xEFBE4786UL,0x0FC19DC6UL,0x240CA1CCUL,
    0x2DE92C6FUL,0x4A7484AAUL,0x5CB0A9DCUL,0x76F988DAUL,
    0x983E5152UL,0xA831C66DUL,0xB00327C8UL,0xBF597FC7UL,
    0xC6E00BF3UL,0xD5A79147UL,0x06CA6351UL,0x14292967UL,
    0x27B70A85UL,0x2E1B2138UL,0x4D2C6DFCUL,0x53380D13UL,
    0x650A7354UL,0x766A0ABBUL,0x81C2C92EUL,0x92722C85UL,
    0xA2BFE8A1UL,0xA81A664BUL,0xC24B8B70UL,0xC76C51A3UL,
    0xD192E819UL,0xD6990624UL,0xF40E3585UL,0x106AA070UL,
    0x19A4C116UL,0x1E376C08UL,0x2748774CUL,0x34B0BCB5UL,
    0x391C0CB3UL,0x4ED8AA4AUL,0x5B9CCA4FUL,0x682E6FF3UL,
    0x748F82EEUL,0x78A5636FUL,0x84C87814UL,0x8CC70208UL,
    0x90BEFFFAUL,0xA4506CEBUL,0xBEF9A3F7UL,0xC67178F2UL
};

static uint32_t ReadBe32(const uint8_t *source)
{
    return ((uint32_t)source[0] << 24) | ((uint32_t)source[1] << 16) |
           ((uint32_t)source[2] << 8) | (uint32_t)source[3];
}

static void WriteBe32(uint8_t *destination, uint32_t value)
{
    destination[0] = (uint8_t)(value >> 24);
    destination[1] = (uint8_t)(value >> 16);
    destination[2] = (uint8_t)(value >> 8);
    destination[3] = (uint8_t)value;
}

static void Sha256_Transform(Sha256Context *context)
{
    uint32_t words[64];
    uint32_t a,b,c,d,e,f,g,h;
    uint32_t index;
    for(index = 0U; index < 16U; index++)
        words[index] = ReadBe32(context->block + index * 4U);
    for(index = 16U; index < 64U; index++)
    {
        uint32_t s0 = ROTR32(words[index - 15U], 7U) ^
                      ROTR32(words[index - 15U], 18U) ^
                      (words[index - 15U] >> 3U);
        uint32_t s1 = ROTR32(words[index - 2U], 17U) ^
                      ROTR32(words[index - 2U], 19U) ^
                      (words[index - 2U] >> 10U);
        words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
    }
    a=context->state[0]; b=context->state[1]; c=context->state[2]; d=context->state[3];
    e=context->state[4]; f=context->state[5]; g=context->state[6]; h=context->state[7];
    for(index = 0U; index < 64U; index++)
    {
        uint32_t sum1 = ROTR32(e,6U) ^ ROTR32(e,11U) ^ ROTR32(e,25U);
        uint32_t choose = (e & f) ^ ((~e) & g);
        uint32_t temporary1 = h + sum1 + choose + sha256Constants[index] + words[index];
        uint32_t sum0 = ROTR32(a,2U) ^ ROTR32(a,13U) ^ ROTR32(a,22U);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temporary2 = sum0 + majority;
        h=g; g=f; f=e; e=d+temporary1; d=c; c=b; b=a; a=temporary1+temporary2;
    }
    context->state[0]+=a; context->state[1]+=b; context->state[2]+=c; context->state[3]+=d;
    context->state[4]+=e; context->state[5]+=f; context->state[6]+=g; context->state[7]+=h;
}

void Sha256_Init(Sha256Context *context)
{
    static const uint32_t initial[8] = {
        0x6A09E667UL,0xBB67AE85UL,0x3C6EF372UL,0xA54FF53AUL,
        0x510E527FUL,0x9B05688CUL,0x1F83D9ABUL,0x5BE0CD19UL
    };
    memcpy(context->state, initial, sizeof(initial));
    context->bit_length = 0U;
    context->block_length = 0U;
}

void Sha256_Update(Sha256Context *context, const uint8_t *data, uint32_t length)
{
    uint32_t index;
    for(index = 0U; index < length; index++)
    {
        context->block[context->block_length++] = data[index];
        if(context->block_length == 64U)
        {
            Sha256_Transform(context);
            context->bit_length += 512U;
            context->block_length = 0U;
        }
    }
}

void Sha256_Final(Sha256Context *context, uint8_t digest[32])
{
    uint32_t index = context->block_length;
    uint64_t totalBits;
    context->block[index++] = 0x80U;
    if(index > 56U)
    {
        while(index < 64U) context->block[index++] = 0U;
        Sha256_Transform(context);
        index = 0U;
    }
    while(index < 56U) context->block[index++] = 0U;
    totalBits = context->bit_length + (uint64_t)context->block_length * 8U;
    for(index = 0U; index < 8U; index++)
        context->block[63U - index] = (uint8_t)(totalBits >> (index * 8U));
    Sha256_Transform(context);
    for(index = 0U; index < 8U; index++) WriteBe32(digest + index * 4U, context->state[index]);
    memset(context, 0, sizeof(*context));
}

void HmacSha256(const uint8_t *key, uint32_t key_length,
                const uint8_t *data, uint32_t data_length,
                uint8_t digest[32])
{
    uint8_t normalized[64];
    uint8_t innerDigest[32];
    uint8_t pad[64];
    Sha256Context context;
    uint32_t index;
    memset(normalized, 0, sizeof(normalized));
    if(key_length > sizeof(normalized))
    {
        Sha256_Init(&context);
        Sha256_Update(&context, key, key_length);
        Sha256_Final(&context, normalized);
    }
    else memcpy(normalized, key, key_length);
    for(index = 0U; index < sizeof(pad); index++) pad[index] = normalized[index] ^ 0x36U;
    Sha256_Init(&context);
    Sha256_Update(&context, pad, sizeof(pad));
    Sha256_Update(&context, data, data_length);
    Sha256_Final(&context, innerDigest);
    for(index = 0U; index < sizeof(pad); index++) pad[index] = normalized[index] ^ 0x5CU;
    Sha256_Init(&context);
    Sha256_Update(&context, pad, sizeof(pad));
    Sha256_Update(&context, innerDigest, sizeof(innerDigest));
    Sha256_Final(&context, digest);
    memset(normalized, 0, sizeof(normalized));
    memset(innerDigest, 0, sizeof(innerDigest));
    memset(pad, 0, sizeof(pad));
}
