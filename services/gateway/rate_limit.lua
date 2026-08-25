-- NOT ACTIVE: loaded by nginx.conf (access_by_lua_file) once the gateway
-- is deployed on AKS. No Python code imports or runs this.
-- Edge rate limiting, run inside the gateway (Nginx/OpenResty or Kong).
-- Stops abusive traffic BEFORE it reaches the Python API: the same
-- INCR-then-EXPIRE counter as the app-level limiter, but keyed by client
-- address, spending gateway cycles instead of API cycles.
-- Takes effect when the gateway is deployed (AKS phase); the API keeps
-- its own per-user limiter as the second layer.

local redis = require "resty.redis"
local red = redis:new()
red:set_timeout(100)

local ok, err = red:connect("redis", 6379)
if not ok then
    -- A dead Redis must not take the API down with it: let traffic pass.
    return
end

local key = "rate_limit:" .. ngx.var.remote_addr
local limit = 100 -- placeholder; set from measured p99 request/min per IP
-- once traffic exists

local current = red:incr(key)
if current == 1 then
    red:expire(key, 60)
end

if current > limit then
    ngx.status = 429
    ngx.say("Rate limit exceeded.")
    return ngx.exit(429)
end
