CREATE DATABASE database_name ENCODING UTF8;
create user myuser with encrypted password 'mypass';
grant all privileges on database mydb to myuser;

                                 ALTER SYSTEM SET enable_full_json TO TRUE;
ALTER SYSTEM SET enable_full_funcscan TO TRUE;
ALTER SYSTEM SET enable_full_lateral_join TO TRUE;
select pg_reload_conf();
show enable_full_funcscan;
